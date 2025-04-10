from data_utils import AnnotationsReader, AnnotationLine
import json


class FashionIQAnnotations(AnnotationsReader):
    def __init__(self, annotations_path, split_path):
        super().__init__(annotations_path, split_path)
        self.lines = self.readlines()
        return

    def readlines(self):
        # create lines from file
        annotations_file = open(self.annotations_path, "r")
        annotations = json.load(annotations_file)
        annotations_file.close()
        lines = []
        if self.split_path:
            split_file = open(self.split_path)
            split_file = open(self.split_path)
            self.split_images = json.load(split_file)
            split_file.close()

        for line in annotations:
            try:
                annotation_line = self.read(line)
            except:
                print(f"Line {line} cannot be parsed, skipping...")
                continue

            if annotation_line:
                lines.append(annotation_line)
        return lines

    def read(self, line):
        # read and parse lines, return AnnotationLine object
        reference_image_name = line["candidate"] + ".png"
        caption = [line["captions"][0] + " and " + line["captions"][1]]

        if not (reference_image_name.removesuffix(".png") in self.split_images):
            return None

        target_image_list = [line["target"] + ".png"]
        return AnnotationLine(reference_image_name, target_image_list, caption)
