# from pandas.tests.frame.methods.test_replace import mix_ab
from src.utils import loss
from src.utils.utils import *
from tqdm import tqdm
def test_image_transform(resize_size=256, crop_size=224, alexnet=False):
    """
    transform for test images
    input:
        resize_size: the size of the resized image
        crop_size: the size of the cropped image
        alexnet: whether to use alexnet normalization
    output:
        transform: the transform object for test images
    """
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # else:
    # normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose(
            [transforms.Resize((resize_size, resize_size)), transforms.CenterCrop(crop_size),
             transforms.ToTensor(), normalize])


def train_image_transform(resize_size=256, crop_size=224, alexnet=False):
    """
    transform for training images
    input:
        resize_size: the size of the resized image
        crop_size: the size of the cropped image
        alexnet: whether to use alexnet normalization
    output:
        transform: the transform object for training images
    """
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # else:
    # normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose(
            [transforms.Resize((resize_size, resize_size)), transforms.RandomCrop(crop_size),
             transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    """
    learning rate scheduler
    input:
        optimizer: the optimizer for updating parameters
        iter_num: current iteration number
        max_iter: total number of iterations
        gamma: decay rate
        power: decay power
    output:
        optimizer: the updated optimizer
    """
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def cal_acc(loader, model_parts, vp=None, text_inputs=None):
    """
    calculate the accuracy of the model
    input:
        loader: the dataloader of the target dataset
        vp: the visual prompt module of the global/local model
        text_inputs: the tokenized text inputs for CLIP
        model_parts: the parts of the global/local model
    output:
        accuracy: the accuracy of the model
        mean_ent: the mean entropy of the model
    """
    # print("Calculating accuracy...")
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if vp is not None:
                inputs = vp(inputs)
            if isinstance(model_parts, (list, tuple)):
                try:
                    local_backbone, local_bottleneck, local_head = model_parts[0], model_parts[1], model_parts[2]
                except:
                    raise ValueError(
                            f"Invalid input when calculating accuracy. Expected 1 or 3 model parts, but got {len(model_parts)}")
                features = local_bottleneck(local_backbone(inputs))
                outputs = local_head(features)  # shape=(1*class_num)
            else:
                global_model = model_parts
                try:
                    # print(f"inputshape:{inputs.shape}, textinputshape:{text_inputs.shape}")
                    outputs, _ = global_model(inputs, text_inputs)  # shape=(1*class_num)
                except:
                    print(f"error occurred, inputshape:{inputs.shape}")
            if i == 0:
                all_output = outputs.float().cpu()
                all_label = labels.float()
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()),
                                       0)  # shape=(n*class_num)
                all_label = torch.cat((all_label, labels.float()), 0)  # shape=(n,)

    _, predict = torch.max(all_output, 1)  # shape=(n,1)
    predict = torch.squeeze(predict).float()  # shape=(n,)
    accuracy = torch.sum(predict == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    return accuracy * 100, mean_ent


def optimizer_copy(optimizer):
    """
    copy the optimizer for updating parameters
    input:
        optimizer: the optimizer for updating parameters
    output:
        optimizer: the copied optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def print_cfg(cfg):
    """
    print the config file
    input:
        cfg: config file
    output:
        s: the string of the config file
    """
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += f"{arg}:{content}\n"
    return s


def obtain_label(loader, local_backbone, local_bottleneck, local_head, local_vp, text_inputs, global_model,
                 global_vp):
    """
    obtain the pseudo-label for the target dataset
    input:
        loader: the dataloader of the target dataset
        local_backbone: the backbone network of the local model
        local_bottleneck: the bottleneck layer of the local model
        local_head: the classification head of the local model
        local_vp
        text_inputs: the tokenized text inputs for CLIP
        global_model: the CLIP model
        global_vp: the visual prompt module of the global model
    output:
        confidence_images: the list of the image paths
        confidence_distribution: the confidence distribution of the pseudo-label
        global_predict: the predicted pseudo-label of the global model
    """

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            local_outputs = local_head(local_bottleneck(local_backbone(local_vp(inputs))))
            global_outputs, _ = global_model(global_vp(inputs), text_inputs)
            # print(f"error occurred, input shape with vp = {global_vp(inputs).shape}, without vp = {inputs.shape}")
            local_outputs = local_outputs.float().cpu()
            global_outputs = global_outputs.float().cpu()
            labels = labels.float().cpu()
            if i == 0:
                all_local_output = local_outputs
                all_global_output = global_outputs
                all_label = labels
            else:
                all_local_output = torch.cat((all_local_output, local_outputs), 0)
                all_global_output = torch.cat((all_global_output, global_outputs), 0)
                all_label = torch.cat((all_label, labels), 0)

    all_global_output = nn.Softmax(dim=1)(all_global_output).cpu()  # shape=(n,class_num)
    _, global_predict = torch.max(all_global_output, 1)
    global_predict = torch.squeeze(global_predict).float()
    global_accuracy = torch.sum(global_predict == all_label).item() / float(all_label.size()[0])

    all_local_output = nn.Softmax(dim=1)(all_local_output).cpu()
    _, local_predict = torch.max(all_local_output, 1)
    local_predict = torch.squeeze(local_predict).float()
    local_accuracy = torch.sum(local_predict == all_label).item() / float(all_label.size()[0])

    all_mix_output = (all_local_output + all_global_output) / 2
    confidence_distribution = all_mix_output.detach()
    confidence_images = loader.dataset.imgs

    return confidence_images, confidence_distribution, all_global_output, local_accuracy, global_accuracy
