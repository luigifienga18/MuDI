import os
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from dreamsim.dreamsim import dreamsim

# Funzione per caricare le immagini di query
def load_query_image(query_dict):
    query_path = query_dict['query_path']
    query_img = []
    for dir_path in query_path:
        imgs = []
        img_paths = os.listdir(dir_path)
        img_paths.sort()
        for img_path in img_paths:
            path = os.path.join(dir_path, img_path)
            if not path.endswith(('.png', '.jpg')):
                continue
            if 'mask' in path:
                continue
            img = Image.open(path)
            img = img.convert('RGB')
            imgs.append(img)
        query_img.append(imgs)
    query_dict['query_img'] = query_img
    return query_dict

# Funzione per il crop e padding dell'immagine
def crop_and_pad_image(image, bbox):
    img_width, img_height = image.size
    left, upper, right, lower = bbox
    cropped_image = image.crop((left, upper, right, lower))
    cropped_width, cropped_height = cropped_image.size
    longer_side = max(cropped_width, cropped_height)
    final_size = (longer_side, longer_side)
    padded_image = Image.new("RGB", final_size, (255, 255, 255))
    x_offset = (longer_side - cropped_width) // 2
    y_offset = (longer_side - cropped_height) // 2
    padded_image.paste(cropped_image, (x_offset, y_offset))
    return padded_image

class eval_with_groundingdino:
    def __init__(self, device='cuda'):
        self.device = device
        # Inizializza il processore e il modello Grounding DINO
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
        # Inizializza DreamSim
        self.ds_model, self.ds_preprocess = dreamsim(pretrained=True, device=device)

    def cache_query_embedding(self, query_dict):
        query_img = query_dict['query_img']
        query_emb = []
        for imgs in query_img:
            embs = []
            for img in imgs:
                emb = self.ds_model.embed(self.ds_preprocess(img).to(self.device))
                embs.append(emb.to('cpu'))
            query_emb.append(torch.cat(embs, dim=0))
        query_dict['query_emb'] = query_emb
        return query_dict

    def query_dict_update(self, query_dict):
        query_dict = load_query_image(query_dict)
        return self.cache_query_embedding(query_dict)

    def compute_distance(self, candidate, query_dict):
        results = {}
        candidate_emb = self.ds_model.embed(self.ds_preprocess(candidate).to(self.device))
        for i, query_emb in enumerate(query_dict['query_emb']):
            distance = 1 - F.cosine_similarity(candidate_emb, query_emb.to(self.device), dim=-1)
            distance = distance.tolist()
            results[i] = distance
        return results

    def groundingdino_distance(self, img, query_dict, box_threshold=0.3, text_threshold=0.25):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        categories = query_dict['query_name']
        # Prepara l'input per Grounding DINO
        inputs = self.processor(images=img, text=categories, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Post-processamento per ottenere le bbox
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[img.size[::-1]]
        )[0]
        boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
        results_list = []
        dreamsim_out = {}
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            candidate = crop_and_pad_image(img, (x1, y1, x2, y2))
            distance = self.compute_distance(candidate, query_dict)
            results_list.append({
                "image": candidate,
                "bbox": [x1, y1, x2, y2],
                "label": categories[label],
                "score": score.item(),
                "distance": distance
            })
        return results_list

    @torch.no_grad()
    def score(self, image, query_dict, box_threshold=0.3, text_threshold=0.25, return_round=False):
        if 'query_emb' not in query_dict:
            query_dict = self.query_dict_update(query_dict)
        results = self.groundingdino_distance(image, query_dict, box_threshold, text_threshold)
        scores = []
        scores_ = []
        for result in results:
            distance = result['distance']
            distance_score = [[1 - x for x in y] for y in distance.values()]
            distance_score_ = [[round(1 - x, 2) for x in y] for y in distance.values()]
            scores.append(distance_score)
            scores_.append(distance_score_)
        return scores_ if return_round else scores
