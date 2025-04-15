% 基础路径
base_path = 'G:\dataset\manifest-1701387492049\LIDC-IDRI';
new_base_path = 'G:\forschung_dataset\LIDC-IDRI';

% 获取所有子目录
subdirs = dir(fullfile(base_path, 'LIDC-IDRI-*'));
subdirs = subdirs([subdirs.isdir]);  % 仅选取目录

% 遍历每个子目录
for i = 1:length(subdirs)
    subdir = subdirs(i).name;
    
    % 完整的子目录路径
    full_subdir_path = fullfile(base_path, subdir);

    % 查找所有.dcm文件
    dcm_files = dir(fullfile(full_subdir_path, '**', '*.dcm'));
    for j = 1:length(dcm_files)
        old_file_path = fullfile(dcm_files(j).folder, dcm_files(j).name);
        new_file_path = fullfile(new_base_path, subdir, dcm_files(j).name);

        % 创建新目录（如果不存在）
        if ~exist(fileparts(new_file_path), 'dir')
            mkdir(fileparts(new_file_path));
        end

        % 移动文件
        movefile(old_file_path, new_file_path);

        % 显示移动信息
        fprintf('Moved: %s -> %s\n', old_file_path, new_file_path);
    end
end