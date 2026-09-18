$ErrorActionPreference = "Continue"
Set-Location 'D:\项目\基于D2NN\simulator'
$env:PYTHONPATH = "D:\项目\基于D2NN\simulator"
Write-Output '=== baseline_d2nn seed42 ==='
& 'D:\项目\基于D2NN\.venv\Scripts\python.exe' 'D:\项目\基于D2NN\simulator\evaluate_clean.py' --checkpoint 'D:\项目\基于D2NN\artifacts\formal_mnist_v2\best_mnist.formal_mnist_v2_baseline_d2nn_seed42.pth' --method-id 'baseline_d2nn' --protocol 'D:\项目\基于D2NN\FORMAL_EXPERIMENT_PROTOCOL_V2.json' --output-dir 'D:\项目\基于D2NN\results\formal_mnist_v2\clean\baseline_d2nn_seed42' --batch-size 128 --cpu-threads 2 --warmup-repeats 1 --measurement-repeats 5 *> 'D:\项目\基于D2NN\results\formal_mnist_v2\logs\local_eval_baseline_d2nn_seed42.log'
Write-Output '    exit=' + $LASTEXITCODE
Write-Output '=== baseline_d2nn seed46 ==='
& 'D:\项目\基于D2NN\.venv\Scripts\python.exe' 'D:\项目\基于D2NN\simulator\evaluate_clean.py' --checkpoint 'D:\项目\基于D2NN\artifacts\formal_mnist_v2\best_mnist.formal_mnist_v2_baseline_d2nn_seed46.pth' --method-id 'baseline_d2nn' --protocol 'D:\项目\基于D2NN\FORMAL_EXPERIMENT_PROTOCOL_V2.json' --output-dir 'D:\项目\基于D2NN\results\formal_mnist_v2\clean\baseline_d2nn_seed46' --batch-size 128 --cpu-threads 2 --warmup-repeats 1 --measurement-repeats 5 *> 'D:\项目\基于D2NN\results\formal_mnist_v2\logs\local_eval_baseline_d2nn_seed46.log'
Write-Output '    exit=' + $LASTEXITCODE
Write-Output '=== robust_d2nn seed45 ==='
& 'D:\项目\基于D2NN\.venv\Scripts\python.exe' 'D:\项目\基于D2NN\simulator\evaluate_clean.py' --checkpoint 'D:\项目\基于D2NN\artifacts\formal_mnist_v2\best_mnist.formal_mnist_v2_robust_d2nn_seed45.pth' --method-id 'robust_d2nn' --protocol 'D:\项目\基于D2NN\FORMAL_EXPERIMENT_PROTOCOL_V2.json' --output-dir 'D:\项目\基于D2NN\results\formal_mnist_v2\clean\robust_d2nn_seed45' --batch-size 128 --cpu-threads 2 --warmup-repeats 1 --measurement-repeats 5 *> 'D:\项目\基于D2NN\results\formal_mnist_v2\logs\local_eval_robust_d2nn_seed45.log'
Write-Output '    exit=' + $LASTEXITCODE
Write-Output '=== hybrid seed44 ==='
& 'D:\项目\基于D2NN\.venv\Scripts\python.exe' 'D:\项目\基于D2NN\simulator\evaluate_clean.py' --checkpoint 'D:\项目\基于D2NN\artifacts\formal_mnist_v2\best_mnist.formal_mnist_v2_hybrid_seed44.pth' --method-id 'hybrid' --protocol 'D:\项目\基于D2NN\FORMAL_EXPERIMENT_PROTOCOL_V2.json' --output-dir 'D:\项目\基于D2NN\results\formal_mnist_v2\clean\hybrid_seed44' --batch-size 128 --cpu-threads 2 --warmup-repeats 1 --measurement-repeats 5 *> 'D:\项目\基于D2NN\results\formal_mnist_v2\logs\local_eval_hybrid_seed44.log'
Write-Output '    exit=' + $LASTEXITCODE
Write-Output '=== hybrid_qat seed43 ==='
& 'D:\项目\基于D2NN\.venv\Scripts\python.exe' 'D:\项目\基于D2NN\simulator\evaluate_clean.py' --checkpoint 'D:\项目\基于D2NN\artifacts\formal_mnist_v2\best_mnist.formal_mnist_v2_hybrid_qat_seed43.pth' --method-id 'hybrid_qat' --protocol 'D:\项目\基于D2NN\FORMAL_EXPERIMENT_PROTOCOL_V2.json' --output-dir 'D:\项目\基于D2NN\results\formal_mnist_v2\clean\hybrid_qat_seed43' --batch-size 128 --cpu-threads 2 --warmup-repeats 1 --measurement-repeats 5 *> 'D:\项目\基于D2NN\results\formal_mnist_v2\logs\local_eval_hybrid_qat_seed43.log'
Write-Output '    exit=' + $LASTEXITCODE
Write-Output '=== lenet5 seed42 ==='
& 'D:\项目\基于D2NN\.venv\Scripts\python.exe' 'D:\项目\基于D2NN\simulator\evaluate_clean.py' --checkpoint 'D:\项目\基于D2NN\artifacts\formal_mnist_v2\best_mnist.formal_mnist_v2_lenet5_seed42.pth' --method-id 'lenet5' --protocol 'D:\项目\基于D2NN\FORMAL_EXPERIMENT_PROTOCOL_V2.json' --output-dir 'D:\项目\基于D2NN\results\formal_mnist_v2\clean\lenet5_seed42' --batch-size 128 --cpu-threads 2 --warmup-repeats 1 --measurement-repeats 5 *> 'D:\项目\基于D2NN\results\formal_mnist_v2\logs\local_eval_lenet5_seed42.log'
Write-Output '    exit=' + $LASTEXITCODE
Write-Output '=== lenet5 seed46 ==='
& 'D:\项目\基于D2NN\.venv\Scripts\python.exe' 'D:\项目\基于D2NN\simulator\evaluate_clean.py' --checkpoint 'D:\项目\基于D2NN\artifacts\formal_mnist_v2\best_mnist.formal_mnist_v2_lenet5_seed46.pth' --method-id 'lenet5' --protocol 'D:\项目\基于D2NN\FORMAL_EXPERIMENT_PROTOCOL_V2.json' --output-dir 'D:\项目\基于D2NN\results\formal_mnist_v2\clean\lenet5_seed46' --batch-size 128 --cpu-threads 2 --warmup-repeats 1 --measurement-repeats 5 *> 'D:\项目\基于D2NN\results\formal_mnist_v2\logs\local_eval_lenet5_seed46.log'
Write-Output '    exit=' + $LASTEXITCODE
Write-Output 'WORKER 0 DONE'

