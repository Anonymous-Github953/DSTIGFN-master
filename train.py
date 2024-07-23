import os
import time
import copy
import torch
import random
import logging
import argparse
import numpy as np
import pandas as pd
import configparser
from lib.engine import trainer
import lib.utils as utils
from datetime import datetime
from lib.data_processing import data_processing, setup_logging


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

def main(args, logger):
    # Initialise seed
    seed_it(6666)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device
    logger.info(f"CUDA: {USE_CUDA}, {device}")
    train_loader, valid_loader, test_loader, scaler = data_processing(args, logger)

    loss = 9999999
    test_log = 999999
    update_flag = 0
    start_epoch = args.start_epoch
    val_time = []
    train_time = []
    train_result = []
    test_result = []

    engine = trainer(scaler, args.in_channels, args.num_nodes, args.channels, args.dropout, args.learning_rate, args.weight_decay, device,
                    args.steps_per_day, args.alph, args.gama, logger)

    # Determining whether to breakpoint training
    if 0 < start_epoch < args.epochs:
        engine.model.load_state_dict(torch.load(args.save + "/train_model_{}.pth".format(start_epoch)))

    for i in range(start_epoch, args.epochs):
        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []

        t1 = time.time()
        for iter, (x, y) in enumerate(train_loader):
            trainx = x.to(device)
            trainx = trainx.transpose(1, 3)
            trainy = y.to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_wmape.append(metrics[3])
        t2 = time.time()

        train_time.append(t2 - t1)

        meam_train_loss = np.mean(train_loss)
        meam_train_mape = np.mean(train_mape)
        meam_train_wmape = np.mean(train_wmape)
        meam_train_rmse = np.mean(train_rmse)

        # Validation Set
        valid_loss = []
        valid_mape = []
        valid_wmape = []
        valid_rmse = []

        v1 = time.time()
        for iter, (x, y) in enumerate(valid_loader):
            testx = x.to(device)
            testx = testx.transpose(1, 3)
            testy = y.to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_wmape.append(metrics[3])
        v2 = time.time()

        val_time.append(v2-v1)

        meam_valid_loss = np.mean(valid_loss)
        meam_valid_mape = np.mean(valid_mape)
        meam_valid_wmape = np.mean(valid_wmape)
        meam_valid_rmse = np.mean(valid_rmse)

        logger.info("Epoch: {:03d}, Training MAE: {:.4f}, Training MAPE: {:.4f}, Training RMSE: {:.4f}, Training WMAPE: {:.4f}, Training Time: {:.4f} secs".format(
                i, meam_train_loss, meam_train_mape, meam_train_rmse, meam_train_wmape, t2 - t1))
        logger.info("Epoch: {:03d}, Validation MAE: {:.4f}, Validation MAPE: {:.4f}, Validation RMSE: {:.4f}, Validation WMAPE: {:.4f}, Validation Time: {:.4f} secs".format(
                i, meam_valid_loss, meam_valid_mape, meam_valid_rmse, meam_valid_wmape, v2 - v1))

        if meam_valid_loss < loss:
            if i < 100:
                logger.info("###Update tasks appear###")
                # It is not necessary to print the results of the test set when epoch is less than 100, because the model has not yet converged.
                loss = meam_valid_loss
                best_model = copy.deepcopy(engine.model.state_dict())
                torch.save(best_model, args.save + "/best_model.pth")

                torch.save(best_model, args.save + "/train_model_{}.pth".format(i))

                update_flag = 0
                logger.info("Updating! Valid Loss:{}, epoch:{}".format(loss, i))
            else:
                # Test Set
                test_loss = []
                test_mape = []
                test_wmape = []
                test_rmse = []

                for iter, (x, y) in enumerate(test_loader):
                    testx = x.to(device)
                    testx = testx.transpose(1, 3)
                    testy = y.to(device)
                    testy = testy.transpose(1, 3)
                    metrics = engine.eval(testx, testy)
                    test_loss.append(metrics[0])
                    test_mape.append(metrics[1])
                    test_rmse.append(metrics[2])
                    test_wmape.append(metrics[3])

                meam_test_loss = np.mean(test_loss)

                if meam_test_loss < test_log:
                    logger.info("###Update tasks appear###")
                    # It is not necessary to print the results of the test set when epoch is less than 100, because the model has not yet converged.
                    test_log = meam_test_loss
                    best_model = copy.deepcopy(engine.model.state_dict())
                    torch.save(best_model, args.save + "/best_model.pth")
                    torch.save(best_model, args.save + "/train_model_{}.pth".format(i))
                    update_flag = 0
                    logger.info("Updating! Test Loss:{}, epoch:{}".format(loss, i))
                else:
                    update_flag += 1
                    logger.info("###Cumulative total of {} No update tasks appear###".format(update_flag))
                    # Determine if you are eligible for an early stop
                    if update_flag >= args.es_patience:
                        break
        else:
            if i >= 100:
                # Test Set
                test_loss = []
                test_mape = []
                test_wmape = []
                test_rmse = []

                for iter, (x, y) in enumerate(test_loader):
                    testx = x.to(device)
                    testx = testx.transpose(1, 3)
                    testy = y.to(device)
                    testy = testy.transpose(1, 3)
                    metrics = engine.eval(testx, testy)
                    test_loss.append(metrics[0])
                    test_mape.append(metrics[1])
                    test_rmse.append(metrics[2])
                    test_wmape.append(metrics[3])

                meam_test_loss = np.mean(test_loss)

                if meam_test_loss < test_log:
                    logger.info("###Update tasks appear###")
                    # It is not necessary to print the results of the test set when epoch is less than 100, because the model has not yet converged.
                    test_log = meam_test_loss
                    best_model = copy.deepcopy(engine.model.state_dict())
                    torch.save(best_model, args.save + "/best_model.pth")
                    torch.save(best_model, args.save + "/train_model_{}.pth".format(i))
                    update_flag = 0
                    logger.info("Updating! Test Loss:{}, epoch:{}".format(loss, i))
                else:
                    update_flag += 1
                    logger.info("###Cumulative total of {} No update tasks appear###".format(update_flag))
                    # Determine if you are eligible for an early stop
                    if update_flag >= args.es_patience:
                        break
            else:
                update_flag += 1
                logger.info("###Cumulative total of {} No update tasks appear###".format(update_flag))
                if update_flag >= args.es_patience:
                    break

        train_m = dict(
            train_loss=meam_train_loss,
            train_rmse=meam_train_rmse,
            train_mape=meam_train_mape,
            train_wmape=meam_train_wmape,
            valid_loss=meam_valid_loss,
            valid_rmse=meam_valid_rmse,
            valid_mape=meam_valid_mape,
            valid_wmape=meam_valid_wmape,
        )

        train_m = pd.Series(train_m)
        train_result.append(train_m)

    logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    logger.info("Average Inference Time: {:.4f} secs/epoch".format(np.mean(val_time)))

    train_csv = pd.DataFrame(train_result)
    train_csv.round(8).to_csv(f"{args.save}/train.csv")

    engine.model.load_state_dict(torch.load(args.save + "/best_model.pth"))
    prediction_values = []
    table_values = []

    for iter, (x, y) in enumerate(test_loader):
        testx = x.to(device)
        testx = testx.transpose(1, 3)
        engine.model.eval()
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        prediction_values.append(preds.squeeze())
        table_values.append(y.transpose(1, 3).squeeze().to(device))

    prediction = torch.cat(prediction_values, dim=0)
    table = torch.cat(table_values, dim=0)

    list_test_mae = []
    list_test_mape = []
    list_test_rmse = []
    list_test_wmape = []

    for k in range(args.future_seq_len):

        pred = scaler.inverse_transform(prediction[:, :, k])
        real = table[:, :, k]
        metrics = utils.metric(pred, real)
        log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
        logger.info(log.format(k, metrics[0], metrics[2], metrics[1], metrics[3]))

        list_test_mae.append(metrics[0])
        list_test_mape.append(metrics[1])
        list_test_rmse.append(metrics[2])
        list_test_wmape.append(metrics[3])

        test_m = dict(
            test_loss=metrics[0],
            test_rmse=metrics[1],
            test_mape=metrics[2],
            test_wmape=metrics[3],
        )
        test_m = pd.Series(test_m)
        test_result.append(test_m)

    log = "On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
    logger.info(log.format(np.mean(list_test_mae), np.mean(list_test_rmse), np.mean(list_test_mape), np.mean(list_test_wmape)))

    test_m = dict(
        test_loss=np.mean(list_test_mae),
        test_rmse=np.mean(list_test_rmse),
        test_mape=np.mean(list_test_mape),
        test_wmape=np.mean(list_test_wmape),
    )

    test_m = pd.Series(test_m)
    test_result.append(test_m)

    test_csv = pd.DataFrame(test_result)
    test_csv.round(8).to_csv(f"{args.save}/test.csv")

    utils.delete_files_with_prefix(args.save, "train_model")

    return list_test_mae, list_test_rmse, list_test_mape, list_test_wmape

if __name__ == "__main__":

    dataSet_name = "PeMS08"
    model_name = "DSTIGFN"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='./config_file/{}_{}.conf'.format(dataSet_name, model_name), type=str,
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
    parser.add_argument("--alph", type=float,
                        default=training_config['alph'], help="alph.")
    parser.add_argument("--gama", type=float,
                        default=training_config['gama'], help="gama.")
    parser.add_argument("--learning_rate", type=float,
                        default=training_config['learning_rate'], help="learning_rate.")
    parser.add_argument("--weight_decay", type=float,
                        default=training_config['weight_decay'], help="weight_decay.")
    parser.add_argument("--es_patience", type=float,
                        default=training_config['es_patience'], help="es_patience.")
    args = parser.parse_args()

    # Create a logger
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = "./experiments/{0}_{1}/{2}".format(dataSet_name, model_name, current_time)
    log_file = "{}-{}-{}.log".format(dataSet_name, model_name, current_time)
    logger = setup_logging(out_dir +"/logs", log_file)
    logger.debug("{} is being trained using {}.".format(model_name, dataSet_name))
    t1 = time.time()
    metric = []
    for i in range(args.runs):
        logger.info("The {} training session is initiated....".format(i+1))

        args.expid = i+1
        args.save = os.path.join(out_dir, "session_" + str(args.expid))
        os.makedirs(args.save, exist_ok=True)
        metric.append(main(args, logger))
        t2 = time.time()
        logger.info("Total time spent: {:.4f}".format(t2 - t1))
        logger.info("The {} training session is complete!".format(i+1))

    metric = np.asarray(metric)
    logger.info(metric)
    for i in range(args.future_seq_len):
        logger.info(f"mae for step{i + 1}: {np.mean(metric[:, 0, i])}±{np.std(metric[:, 0, i])}")
        logger.info(f"mape for step{i + 1}: {np.mean(metric[:, 1, i])}±{np.std(metric[:, 1, i])}")
        logger.info(f"rmse for step{i + 1}: {np.mean(metric[:, 2, i])}±{np.std(metric[:, 2, i])}")
    logger.info(f"mean of best mae: {np.mean(metric[:, 0])}±{np.std(np.mean(metric[:, 0], axis=1))}")
    logger.info(f"mean of best mape: {np.mean(metric[:, 1])}±{np.std(np.mean(metric[:, 1], axis=1))}")
    logger.info(f"mean of best rmse: {np.mean(metric[:, 2])}±{np.std(np.mean(metric[:, 2], axis=1))}")


