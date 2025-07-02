import os
import json
import re

class PaperSolver(object):
    def __init__(self, root='dataset/mypaper-large-only', mode="normal-anomaly"):
        self.root = root
        self.meta_path = os.path.join(root,'meta_all.json')
        self.mode = mode
        self.CLSNAMES = [root.split('/')[-1]]

    def run(self):
        info = dict(train={}, test={})
        normal_samples_train = 0
        anomaly_samples_train = 0
        normal_samples_test = 0
        anomaly_samples_test = 0
        for cls_name in self.CLSNAMES:
            normals = []
            anomalies = []
            masks = []
            normal_path = os.path.join(self.root, "normal", "image")
            anomaly_path = os.path.join(self.root, "anomaly", "image")

            if os.path.exists(normal_path):
                normals += os.listdir(normal_path)
            if os.path.exists(anomaly_path):
                anomalies += os.listdir(anomaly_path)

            if not normals and not anomalies:
                print("No samples found in the dataset.")
                return

            train_info = []
            test_info = []
            for specie in normals+anomalies:
                file_number = specie.split('.')[0]
                is_abnormal = True if specie in anomalies else False
                # 全部用来训练及测试
                img_names = specie
                mask_names = file_number+'.png' if is_abnormal else None
                img_path = os.path.join(self.root, "anomaly" if is_abnormal else "normal", "image", img_names)
                mask_path = os.path.join(self.root, "anomaly", "mask", mask_names)
                info_img = dict(
                    img_path=img_path,
                    mask_path=mask_path if is_abnormal and os.path.isfile(mask_path) else '',
                    cls_name=cls_name,
                    specie_name='abnormal' if is_abnormal else 'good',
                    anomaly=1 if is_abnormal else 0,
                )
                if self.mode == "normal-anomaly" or not is_abnormal:
                    train_info.append(info_img)
                    if is_abnormal:
                        anomaly_samples_train += 1
                    else:
                        normal_samples_train += 1
                test_info.append(info_img)
                if is_abnormal:
                    anomaly_samples_test += 1
                else:
                    normal_samples_test += 1

                # info[phase][cls_name] = train_info
            info['train'][cls_name] =train_info
            info['test'][cls_name] =test_info
        # # 全部用来测试
        # phase = 'train'
        # info[phase][cls_name] = train_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('Train: normal_samples', normal_samples_train, 'anomaly_samples', anomaly_samples_train)
        print('Test: normal_samples', normal_samples_test, 'anomaly_samples', anomaly_samples_test)


if __name__ == '__main__':
    root = r'/remote-home/iot_hanxiang/dataset/mypaper-large-only'
    runner = PaperSolver(root)
    runner.run()

