import argparse
import open_clip
import torch
import os
import json

from annotations_classes.flickr30k_annotations import Flickr30kAnnotations
from annotations_classes.circo_annotations import CircoAnnotations
from annotations_classes.fashioniq_annotations import FashionIQAnnotations
from annotations_classes.mydataset_annotations import MyDatasetAnnotations

from utils import (
    recallAtK,
    retrieveComposed,
    retrieveTextToImage,
    meanAveragePrecisionAtK,
)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset",
        required=True,
        choices=["flickr30k", "fashioniq", "circo", "mydataset"],
    )
    argparser.add_argument("--features_path", required=True)
    argparser.add_argument("--textual_embedding_left")
    argparser.add_argument("--textual_embedding_right")
    argparser.add_argument("--annotations_path", required=True)
    argparser.add_argument("--split_file_path")
    argparser.add_argument(
        "--retrieval_type", choices=["composed", "text-to-image"], required=True
    )
    argparser.add_argument("--model_name", required=True)
    argparser.add_argument("--pretraining", required=True)
    argparser.add_argument(
        "--metric", choices=["recall_at_k", "mAP_at_k"], default="recall_at_k"
    )

    args = argparser.parse_args()

    annotations_path = args.annotations_path

    annotations = None
    if args.dataset == "flickr30k":
        split_path = args.split_file_path
        annotations = Flickr30kAnnotations(annotations_path, split_path)
    if args.dataset == "circo":
        annotations = CircoAnnotations(annotations_path)
    if args.dataset == "fashioniq":
        split_path = args.split_file_path
        annotations = FashionIQAnnotations(annotations_path, split_path)
    if args.dataset == "mydataset":
        annotations = MyDatasetAnnotations(annotations_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelName = args.model_name
    pretraining = args.pretraining
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=modelName, pretrained=pretraining
    )
    model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(modelName)

    feature_dict = torch.load(
        os.path.join(args.features_path),
        map_location=device,
    )

    K_list = [1, 2, 5, 10, 25, 50]
    sum_for_k = dict((k, 0) for k in K_list)

    print("# of annotations: ", len(annotations))

    for i in range(len(annotations)):
        reference_image_name = annotations[i].reference_image_name
        caption = annotations[i].caption
        target_image_list = annotations[i].target_image_list

        print(i, reference_image_name, caption, target_image_list)

        if args.retrieval_type == "text-to-image":
            output_ids = retrieveTextToImage(
                max(K_list), caption, feature_dict, model, tokenizer, device
            )

        if args.retrieval_type == "composed":
            reference_image_features = feature_dict[reference_image_name]
            output_ids = retrieveComposed(
                max(K_list),
                reference_image_features,
                caption,
                feature_dict,
                model,
                tokenizer,
                device,
            )

        output_ids = [x[1] for x in output_ids]
        print(output_ids)

        for k in K_list:
            if args.metric == "recall_at_k":
                sum_for_k[k] += recallAtK(k, target_image_list, output_ids)
            if args.metric == "mAP_at_k":
                sum_for_k[k] += meanAveragePrecisionAtK(
                    k, target_image_list, output_ids
                )
        print(sum_for_k)
    for k in K_list:
        sum_for_k[k] /= len(annotations)

    print(f"Results: ({args.metric})", sum_for_k)


if __name__ == "__main__":
    main()
