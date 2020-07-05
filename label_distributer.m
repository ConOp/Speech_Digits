function L = label_distributer(datastore)

files_num = numel(datastore.Files)
tmp = cell(files_num,1);
for file = 1:numel(datastore.Files)
    index = regexp(datastore.Files{file},"[0-9]+_");
    tmp{file} = datastore.Files{file}(index);
end
L = categorical(tmp);
end