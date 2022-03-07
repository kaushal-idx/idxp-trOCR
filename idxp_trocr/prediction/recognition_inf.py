from ..data.recognition_data import RecognitionDataset
from torch.utils.data import DataLoader
import torch



def main(inp, trocr_processor, trocr_model, config):
    rec_dataset = RecognitionDataset(inp, trocr_processor)
    rec_dataloader = DataLoader(rec_dataset, batch_size = config["rec_batch_size"], num_workers=config["num_workers"])
    result = []

    device = "cuda" if config["cuda"] else "cpu"

    for i,batch in enumerate(rec_dataloader):
        with torch.no_grad():
            pixel_values = batch["pixel_values"].to(device)
            idxs = batch["idxs"]
            left = batch["left"]
            right = batch["right"]
            top = batch["top"]
            bottom = batch["bottom"]
            word_num = batch["word_num"]
            
            outputs = trocr_model.generate(pixel_values,
                                    num_beams=3, 
                                    no_repeat_ngram_size=2, 
                                    num_return_sequences=1, 
                                    early_stopping=True,
                                    output_scores=True, 
                                    return_dict_in_generate=True,
                                    max_length=22)
            # get conf score
            tuple_of_logits = torch.stack(outputs.scores, dim=0)
            logits = torch.einsum("mbv->bmv",tuple_of_logits)
            soft_logits = torch.nn.functional.softmax(logits, dim=-1)
            probs = torch.amax(soft_logits, dim=-1)
            batch_probs = torch.mean(probs, dim=-1)
            probs_for_each_image = batch_probs.reshape(len(idxs.tolist()), -1)
            probs_first_sequence = probs_for_each_image[:,0]

            # decode
            pred_str = trocr_processor.batch_decode(outputs.sequences, skip_special_tokens=True)
            
            result.append({
                "word_num": word_num.tolist(),
                "left":left.tolist(),
                "right":right.tolist(),
                "top":top.tolist(),
                "bottom":bottom.tolist(),
                "text":pred_str,
                "conf":probs_first_sequence.tolist(),
                "idxs":idxs.tolist(),
            })
    return result
