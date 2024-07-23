import os
import torch


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

# def MAPE_torch(pred, true, mask_value=None):
#     with torch.no_grad():
#         if mask_value is not None:
#             mask = true > mask_value
#             pred = torch.masked_select(pred, mask)
#             true = torch.masked_select(true, mask)
#
#         # Handle zero values in true to avoid division by zero
#         epsilon = 1e-10  # small value to avoid division by zero
#         true = torch.where(true == 0, torch.tensor(epsilon, device=true.device), true)
#
#         mape = torch.abs((true - pred) / true)
#         mape = torch.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0)
#
#         # Cap MAPE values to reduce the impact of outliers
#         mape = torch.clamp(mape, max=100)
#
#         return torch.mean(mape)

def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def WMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    loss = torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))
    return loss


def metric(pred, real):
    mae = MAE_torch(pred, real, 0.0).item()
    mape = MAPE_torch(pred, real, 0.0).item()
    wmape = WMAPE_torch(pred, real, 0.0).item()
    rmse = RMSE_torch(pred, real, 0.0).item()
    return mae, mape, rmse, wmape


def delete_files_with_prefix(directory, prefix):
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")
