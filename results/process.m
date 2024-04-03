filtered_files = dir('fedXGboost_300_*.mat');

% Filter files containing 'experiment' in their names and bigger than 500Kb
%filtered_files = files(cellfun(@contains, {files.name}, 'fedXGboost_70'));
% Access filenames and load data
filenames = {filtered_files.name};
for i = 1:length(filenames)
  data = load(filenames{i});
  acc_clients_xg(:,i)= data.Accuracy_clients;
  acc_federation_xg(i) = data.Accuracy_federation;
end
avg_acc_clients_xg = mean(mean(acc_clients_xg.'));
avg_acc_fed_xg = mean(acc_federation_xg);

filtered_files = dir('fedavg_300_*.mat');
%filtered_files = files(cellfun(@contains, {files.name}, 'fedavg_70'));

% Access filenames and load data
filenames = {filtered_files.name};
for i = 1:length(filenames)
  data = load(filenames{i});
  acc_clients_fg(:,i)= data.Accuracy_clients;
  acc_federation_fg(i) = data.Accuracy_federation;
end
avg_acc_clients_fg = mean(mean(acc_clients_fg.'));
avg_acc_fed_fg = mean(acc_federation_fg);