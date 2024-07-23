import os
import torch
import logging
import argparse
import numpy as np
import configparser
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def generate_datasets(data, index_list):
    x, y = [], []
    for start, end, future_end in index_list:
        x.append(data[start:end, ...])  # 选择历史数据
        y.append(data[end:future_end, ...])  # 选择未来数据
    return np.array(x), np.array(y)


def setup_logging(log_dir='logs', log_file='data_processing.log'):
    """
    配置日志记录器
    :param log_dir: 日志文件夹路径
    :param log_file: 日志文件名
    :return: 配置好的日志记录器
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def data_processing(args: argparse.Namespace, logger):
    """Preprocess and generate train/valid/test datasets.
    Args:
        args (argparse): configurations of preprocessing
    """
    # print args
    logger.info("-" * (20 + 45 + 5))
    for key, value in sorted(vars(args).items()):
        logger.info("|{0:>20} = {1:<45}|".format(key, str(value)))
    logger.info("-" * (20 + 45 + 5))

    target_channel = [0]
    batch_size = args.batch_size
    data = np.load(args.data_file_path)["data"]
    data = data[..., target_channel]

    # split data
    l, n, f = data.shape
    num_samples = l - (args.history_seq_len + args.future_seq_len) + 1
    train_num = round(num_samples * args.train_ratio)
    valid_num = round(num_samples * args.valid_ratio)
    test_num = num_samples - train_num - valid_num

    index_list = []
    for t in range(args.history_seq_len, num_samples + args.history_seq_len):
        index = (t - args.history_seq_len, t, t + args.future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    valid_index = index_list[train_num: train_num + valid_num]
    test_index = index_list[train_num + valid_num: train_num + valid_num + test_num]
    mean = data[:train_index[-1][1], ...][..., 0].mean()
    std = data[:train_index[-1][1], ...][..., 0].std()

    scaler = StandardScaler(mean, std)

    # add temporal feature
    feature_list = [data]
    if args.time_of_day:
        # numerical time_of_day
        tod = [i % args.steps_per_day /
               args.steps_per_day for i in range(data.shape[0])]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if args.day_of_week:
        # numerical day_of_week
        dow = [(i // args.steps_per_day) % 7 / 7 for i in range(data.shape[0])]
        dow = np.array(dow)
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    processed_data = np.concatenate(feature_list, axis=-1)

    # Cutting and standardisation data
    train_x, train_y = generate_datasets(processed_data, train_index)
    train_x_norm = scaler.transform(train_x[:, :, :, [0]])
    train_x[:, :, :, [0]] = train_x_norm
    # train_y_norm = scaler.transform(train_y[:, :, :, [0]])
    # train_y[:, :, :, [0]] = train_y_norm
    train_y = train_y[:, :, :, [0]]

    valid_x, valid_y = generate_datasets(processed_data, valid_index)
    valid_x_norm = scaler.transform(valid_x[:, :, :, [0]])
    valid_x[:, :, :, [0]] = valid_x_norm
    # valid_y_norm = scaler.transform(valid_y[:, :, :, [0]])
    # valid_y[:, :, :, [0]] = valid_y_norm
    valid_y = valid_y[:, :, :, [0]]

    test_x, test_y = generate_datasets(processed_data, test_index)
    test_x_norm = scaler.transform(test_x[:, :, :, [0]])
    test_x[:, :, :, [0]] = test_x_norm
    test_y = test_y[:, :, :, [0]]

    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_target_tensor = torch.from_numpy(train_y).type(torch.FloatTensor)
    train_dataset = TensorDataset(train_x_tensor, train_target_tensor)

    valid_x_tensor = torch.from_numpy(valid_x).type(torch.FloatTensor)
    valid_target_tensor = torch.from_numpy(valid_y).type(torch.FloatTensor)
    valid_dataset = TensorDataset(valid_x_tensor, valid_target_tensor)

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor)
    test_target_tensor = torch.from_numpy(test_y).type(torch.FloatTensor)
    test_dataset = TensorDataset(test_x_tensor, test_target_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    logger.info("train_x:{}, train_y:{}".format(train_x_tensor.size(), train_target_tensor.size()))
    logger.info("valid_x:{}, valid_y:{}".format(valid_x_tensor.size(), valid_target_tensor.size()))
    logger.info("test_x:{}, test_y:{}".format(test_x_tensor.size(), test_target_tensor.size()))

    return train_loader, valid_loader, test_loader, scaler


if __name__ == "__main__":

    dataSet_name = "PeMS08"
    model_name = "DSTIGFN"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='../config_file/{}_{}.conf'.format(dataSet_name, model_name), type=str,
                        help="configuration file path")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']

    parser.add_argument("--data_file_path", type=str,
                        default=data_config['data_file_path'], help="traffic readings.")
    parser.add_argument("--dataset_name", type=str,
                        default=data_config['dataset_name'], help="traffic readings.")
    parser.add_argument("--history_seq_len", type=int,
                        default=data_config['history_seq_len'], help="history Sequence Length.")
    parser.add_argument("--future_seq_len", type=int,
                        default=data_config['future_seq_len'], help="prection Sequence Length.")
    parser.add_argument("--batch_size", type=int,
                        default=training_config['batch_size'], help="batch size.")
    parser.add_argument("--steps_per_day", type=int,
                        default=data_config['steps_per_day'], help="steps per day.")
    parser.add_argument("--time_of_day", type=bool, default=data_config['TOD'],
                        help="Add feature time_of_day.")
    parser.add_argument("--day_of_week", type=bool, default=data_config['DOW'],
                        help="Add feature day_of_week.")
    parser.add_argument("--train_ratio", type=float,
                        default=data_config['train_ratio'], help="Train ratio")
    parser.add_argument("--valid_ratio", type=float,
                        default=data_config['valid_ratio'], help="Validate ratio.")
    parser.add_argument("--norm_each_channel", type=float, help="Validate ratio.")
    parser.add_argument("--runs", type=int,
                        default=training_config['runs'], help="runs.")
    parser.add_argument("--device_id", type=str,
                        default=training_config['device_id'], help="Gpu_device_id.")
    parser.add_argument("--start_epoch", type=int,
                        default=training_config['start_epoch'], help="start_epoch.")
    parser.add_argument("--epochs", type=int,
                        default=training_config['epochs'], help="epochs.")
    parser.add_argument("--num_nodes", type=int,
                        default=data_config['num_of_vertices'], help="num_nodes.")
    parser.add_argument("--in_channels", type=int,
                        default=training_config['in_channels'], help="in_channels.")
    parser.add_argument("--channels", type=int,
                        default=training_config['nb_chev_filter'], help="channels.")
    parser.add_argument("--print_every", type=int,
                        default=training_config['print_every'], help="print_every.")
    parser.add_argument("--dropout", type=float,
                        default=training_config['dropout'], help="dropout.")
    parser.add_argument("--learning_rate", type=float,
                        default=training_config['learning_rate'], help="learning_rate.")
    parser.add_argument("--weight_decay", type=float,
                        default=training_config['weight_decay'], help="weight_decay.")
    parser.add_argument("--es_patience", type=float,
                        default=training_config['es_patience'], help="es_patience.")
    args = parser.parse_args()

    # Create a logger
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = "./experiments/{0}_{1}/{2}/logs".format(dataSet_name, model_name, current_time)
    log_file = "{}-{}-{}.log".format(dataSet_name, model_name, current_time)
    logger = setup_logging(log_dir, log_file)
    logger.info("Program started")
    logger.debug("{} is being trained using {}.".format(model_name, dataSet_name))
    train_loader, valid_loader, test_loader, scaler = data_processing(args, logger)
    logger.info("Program finished")
    # for batch_index, batch_data in enumerate(train_loader):
    #     encoder_inputs, labels = batch_data
    #     print("encoder_inputs:", encoder_inputs.shape)
    #     # labels = scaler.inverse_transform(labels[:,:,:,[0]])
