function [id_data, val_data] = split_data(data, split_idx)
% SPLIT_DATA Splits a dataset into model identification data and model
%   validation data. No shuffling, model identification data first.
%
% Inputs:
% - data: dataset to split, time in the row-wise dimension
% - split_idx: index at which to split
%
% Outputs:
% - id_data: model identification dataset
% - val_data: model validation dataset
%
% Jesse Hagenaars - 31.05.2018

id_data = data(1:split_idx, :);
val_data = data(split_idx + 1:end, :);

end