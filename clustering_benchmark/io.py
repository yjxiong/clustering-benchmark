import json


def read_groundtruth_file_v1(filename, filter_set=None):
    gt_data = json.load(open(filename))
    persons = [x['person'] for x in gt_data.values()]
    person_mapping = {pid: i for i, pid in enumerate(set(persons))}

    if filter_set is not None:
        image2id_mapping = {image_id: person_mapping[image_data['person']] for image_id, image_data in gt_data.items()
                            if image_data['set'] == filter_set}
    else:
        image2id_mapping = {image_id: person_mapping[image_data['person']] for image_id, image_data in gt_data.items()}

    return image2id_mapping
