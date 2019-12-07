for i in {1..4}; do vsub -c "python run.py --mode $i --pin 0 --pout 0 --sep 0 --app 1" --part dpt-gpu-EL7 --name RFI-mode$i --mem 25000; done

vsub -c "python run.py --mode 4 --pin 0 --pout 0 --sep 0 --app 1" --part dpt-gpu-EL7 --name RFI-mode4 --mem 35000

for i in {1..4}; do vsub -c "python run.py --mode 2 --pin 0 --pout 0 --sep $i --app 1" --part dpt-gpu-EL7 --name RFI-pol$i --mem 25000; done

for i in {1..15}; do vsub -c "python run.py --mode 3 --pin 0 --pout 0 --sep $i --app 1" --part dpt-gpu-EL7 --name RFI-bl$i --mem 25000; done

for app in {2..5}; do vsub -c "python run.py --mode 2 --pin 0 --pout 0 --sep 0 --app $app" --part dpt-gpu-EL7 --name RFI-app$app --mem 25000; done

for i in {1..2}; do vsub -c "python run.py --mode $i --pin 1 --pout 0 --sep 0 --app 1" --part dpt-gpu-EL7 --name RFI-m${i}-pi --mem 25000; done

for i in {1..2}; do vsub -c "python run.py --mode $i --pin 0 --pout 1 --sep 0 --app 1" --part dpt-gpu-EL7 --name RFI-m${i}-po --mem 25000; done


############ TEST
#   GPU
for i in {1..3}; do vsub -c "python run_test.py --mode $i --pin 0 --pout 0 --sep 0 --app 1" --part dpt-gpu-EL7 --name RFI-mode$i --mem 25000; done

vsub -c "python run_test.py --mode 4 --pin 0 --pout 0 --sep 0 --app 1" --part dpt-gpu-EL7 --name RFI-mode4 --mem 35000

for i in {1..4}; do vsub -c "python run_test.py --mode 2 --pin 0 --pout 0 --sep $i --app 1" --part dpt-gpu-EL7 --name RFI-pol$i --mem 25000; done

for i in {1..15}; do vsub -c "python run_test.py --mode 3 --pin 0 --pout 0 --sep $i --app 1" --part dpt-gpu-EL7 --name RFI-bl$i --mem 25000; done

for app in {2..5}; do vsub -c "python run_test.py --mode 2 --pin 0 --pout 0 --sep 0 --app $app" --part dpt-gpu-EL7 --name RFI-app$app --mem 25000; done

for i in {1..2}; do vsub -c "python run_test.py --mode $i --pin 1 --pout 0 --sep 0 --app 1" --part dpt-gpu-EL7 --name RFI-m${i}-pi --mem 25000; done

for i in {1..2}; do vsub -c "python run_test.py --mode $i --pin 0 --pout 1 --sep 0 --app 1" --part dpt-gpu-EL7 --name RFI-m${i}-po --mem 25000; done

###   test_per_range

for i in {1..3}; do vsub -c "python test_per_range.py --mode $i --pin 0 --pout 0 --sep 0 --app 1" --part dpt-EL7 --name RFI-mode$i; done

vsub -c "python test_per_range.py --mode 4 --pin 0 --pout 0 --sep 0 --app 1" --part dpt-EL7 --name RFI-mode4

for i in {1..4}; do vsub -c "python test_per_range.py --mode 2 --pin 0 --pout 0 --sep $i --app 1" --part dpt-EL7 --name RFI-pol$i ; done

for i in {1..15}; do vsub -c "python test_per_range.py --mode 3 --pin 0 --pout 0 --sep $i --app 1" --part dpt-EL7 --name RFI-bl$i; done

for app in {2..5}; do vsub -c "python test_per_range.py --mode 2 --pin 0 --pout 0 --sep 0 --app $app" --part dpt-EL7 --name RFI-app$app; done

for i in {1..2}; do vsub -c "python test_per_range.py --mode $i --pin 1 --pout 0 --sep 0 --app 1" --part dpt-EL7 --name RFI-m${i}-pi; done

for i in {1..2}; do vsub -c "python test_per_range.py --mode $i --pin 0 --pout 1 --sep 0 --app 1" --part dpt-EL7 --name RFI-m${i}-po; done





