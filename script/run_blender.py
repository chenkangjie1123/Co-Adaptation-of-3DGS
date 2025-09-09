import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import json

scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
dataset_name = "Blender"
data_base_path='./data/nerf_synthetic'
out_base_path=f'./output/{dataset_name}'
n_views = 8
resolution = 2


# Notice: When training our method on the Blender dataset, we observed that 
# the evaluation metrics vary significantly between using a white background 
# and a black background. In our paper, we adopt the white background setting.
dropout_factor = 0.3 # 0.2 or 0.3
sigma_noise = 0.0 # 0.8



excluded_gpus = set()
mem_threshold = 0.2
base_dir = os.path.abspath(os.getcwd())

jobs = scenes

for scene in scenes:
    model_path1 = f"{out_base_path}/{scene}_{n_views}views"
    os.makedirs(model_path1, exist_ok=True)

def train_block(gpu_id, scene):
       log_file = f"{out_base_path}/{scene}_{n_views}views/log.txt"
       with open(log_file, "w") as f:
              f.write(f"Starting training for scene: {scene} on GPU {gpu_id}\n")
              
       cmd = (
              f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}_{n_views}views --white_background '
              f'--n_views {n_views} --dataset_name {dataset_name} --resolution {resolution} --eval '
              f'--dropout_factor {dropout_factor} --sigma_noise {sigma_noise} --iterations 7000 --shift_cam_start 4000'
              )
       print(cmd)
       os.system(cmd + f" >> {log_file} 2>&1")

       cmd = (f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py --white_background '
              f'--model_path {out_base_path}/{scene}_{n_views}views '
              f'--n_views {n_views} --resolution {resolution} --eval '
              f'--dropout_factor {dropout_factor} '
              f'--dataset_name {dataset_name}'
              )
       print(cmd)
       os.system(cmd + f" >> {log_file} 2>&1")

       cmd = (f'CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py '
              f'--model_path {out_base_path}/{scene}_{n_views}views '
              )
       print(cmd)
       os.system(cmd + f" >> {log_file} 2>&1")

       return True

def worker(gpu_id, scene):
    print(f"Starting job on GPU {gpu_id} with scene {scene}\n")
    train_block(gpu_id, scene)
    print(f"Finished job on GPU {gpu_id} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.


def dispatch_jobs(jobs, executor):
       future_to_job = {}
       reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

       while jobs or future_to_job:
              # Get the list of available GPUs, not including those that are reserved.
              all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxLoad=0.9, maxMemory=mem_threshold))
              available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

              # Launch new jobs on available GPUs
              while available_gpus and jobs:
                     gpu = available_gpus.pop(0)
                     job = jobs.pop(0)
                     future = executor.submit(worker, gpu, job)  # Unpacking job as arguments to worker
                     future_to_job[future] = (gpu, job)
                     reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

              # Check for completed jobs and remove them from the list of running jobs.
              # Also, release the GPUs they were using.
              done_futures = [future for future in future_to_job if future.done()]
              for future in done_futures:
                     job = future_to_job.pop(future)  # Remove the job associated with the completed future
                     gpu = job[0]  # The GPU is the first element in each job tuple
                     reserved_gpus.discard(gpu)  # Release this GPU
                     print(f"Job {job} has finished., rellasing GPU {gpu}")
              # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
              # when there are no GPUs available.
              if len(jobs) > 0:
                     print("No GPU available at the moment. Retrying in 1 minutes.")
                     time.sleep(60)
              else:
                     time.sleep(10)

       print("All blocks have been processed.")


def collect_results(base_dir, output_file):
    """
    遍历 base_dir 下所有子文件夹，将每个子文件夹中存在的 results.json 文件内容读取，
    并将结果以场景文件夹名称为 key 汇总到一个字典中，最后保存到 output_file 中。
    
    参数:
      base_dir: str，包含多个场景子文件夹的目录路径
      output_file: str，保存汇总结果的 JSON 文件路径
    """
    summary = {}
    psnr_avg = 0
    ssim_avg = 0
    lpips_avg = 0
    num = 0
    
    # 遍历 base_dir 下的所有子文件夹
    for scene in os.listdir(base_dir):
        scene_path = os.path.join(base_dir, scene)
        if os.path.isdir(scene_path):
            num += 1
            results_path = os.path.join(scene_path, 'results.json')
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    # 将当前场景的结果存入 summary 字典中，key 为场景名称
                    summary[scene] = results
                    psnr_avg += results["ours_7000"]["PSNR"]
                    ssim_avg += results["ours_7000"]["SSIM"]
                    lpips_avg += results["ours_7000"]["LPIPS"]
                    print(f"读取场景 {scene} 的结果成功。")
                except Exception as e:
                    print(f"读取场景 {scene} 的结果时出错: {e}")
            else:
                print(f"场景 {scene} 中没有找到 results.json 文件。")

    summary["all_avg"] = {
           "PSNR": psnr_avg/num,
           "SSIM": ssim_avg/num,
           "LPIPS": lpips_avg/num
    }
    # 将汇总结果写入到 output_file 中（格式化输出，便于查看）
    try:
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"所有结果已保存到 {output_file}")
    except Exception as e:
        print(f"保存结果到文件 {output_file} 时出错: {e}")

# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)
collect_results(out_base_path, os.path.join(out_base_path, "summary_results.json"))
