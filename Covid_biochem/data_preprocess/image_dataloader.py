from torch.utils.data import Dataset
import os
import cv2


class DataLoaderCXR(Dataset):

    def __init__(self, image_folder, target, df, input_col, img_size):
        self.patient_ids, self.targets = self.get_patient_ids(df, input_col, target)
        self.image_folder = image_folder
        self.img_size = img_size

    def get_patient_ids(self, df, input_col, target):
        patient_ids = df[input_col].values.tolist()
        targets = df[target].values.tolist()
        return patient_ids, targets

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        path_name = os.path.join(self.image_folder, str(self.patient_ids[idx]))
        image_name = os.path.join(path_name, r'*.jpg')
        image = cv2.imread(image_name, 0)
        image = cv2.resize(image, (self.img_size, self.img_size))
        return image, self.targets[idx]
