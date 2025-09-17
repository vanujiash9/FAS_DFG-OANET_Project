with open('src/scripts/run_pipeline.py', 'r') as f:
    content = f.read()

# Thêm multiprocessing setup ở đầu
mp_setup = '''import multiprocessing
import torch.multiprocessing as mp

# Fix multiprocessing for CUDA
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    multiprocessing.set_start_method('spawn', force=True)

'''

# Tìm vị trí sau imports
import_end = content.find('\n\n')
if import_end > 0:
    new_content = content[:import_end] + '\n' + mp_setup + content[import_end:]
    
    with open('src/scripts/run_pipeline.py', 'w') as f:
        f.write(new_content)
    
    print("✓ Added multiprocessing fix")
else:
    print("✗ Could not find import section")
