import torch


def retrieveTextToImage(num, reference_text, feature_dict, model, tokenizer, device):
    with torch.no_grad():  # Disable gradient tracking
        input_text = tokenizer(reference_text).to(device)

        text_features = model.encode_text(input_text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        distance_list = []
        for key, image_features in feature_dict.items():  # Iterate directly over items
            distance = (100.0 * image_features.float() @ text_features.T).item()
            distance_list.append((distance, key))

        distance_list.sort(reverse=True)

        # Clear variables that are no longer needed
        del input_text
        del text_features

        ret_list = distance_list[:num]
        del distance_list  # Clear the distance list

        torch.cuda.empty_cache()  # Free up GPU memory
        return ret_list


def retrieveComposed(
    num, ref_img_features, reference_text, feature_dict, model, tokenizer, device
):
    with torch.no_grad():  # Disable gradient tracking
        input_text = tokenizer(reference_text).to(device)

        text_features = model.encode_text(input_text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        distance_list = []
        for key, image_features in feature_dict.items():  # Iterate directly over items
            distance = (
                100.0
                * image_features.float()
                @ (torch.add(ref_img_features, text_features)).T
            ).item()
            distance_list.append((distance, key))

        distance_list.sort(reverse=True)

        # Clear variables that are no longer needed
        del input_text
        del text_features

        ret_list = distance_list[:num]
        del distance_list  # Clear the distance list

        torch.cuda.empty_cache()  # Free up GPU memory
        return ret_list


import json


def meanAveragePrecisionAtK(K, gt_ids, output_ids):
    sum = 0
    positives_by_k = 0
    for k in range(K):
        if output_ids[k] in gt_ids:  # check positivity at rank k
            positives_by_k = positives_by_k + 1
            sum += positives_by_k / (k + 1)
    return sum / min(K, len(gt_ids))


def recallAtK(K, gt_ids, output_ids):
    found = 0
    for k in range(K):
        if output_ids[k] in gt_ids:
            found += 1
    return found / len(gt_ids)
