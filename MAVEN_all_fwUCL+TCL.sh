source activate zhangchenlong
for i in 1 2 5 10
do
    for j in 5 10
    do
        for k in shuffle
        do
            for l in none ce
            do
                for m in 10 20
                do
                    for n in ACE MAVEN
                    do
                        python train.py \
                            --data-root ./data_incremental \
                            --dataset $n \
                            --backbone bert-base-uncased \
                            --lr 2e-5 \
                            --decay 1e-4 \
                            --no-freeze-bert \
                            --shot-num $j \
                            --batch-size 4 \
                            --device cuda:4 \
                            --log \
                            --log-dir ./outputs/log_incremental/temp7_submax/first_wo_UCL+TCL/ \
                            --log-name a${k}_l${l}_r${i} \
                            --dweight_loss \
                            --rep-aug mean \
                            --distill mul \
                            --epoch 30 \
                            --class-num $m \
                            --single-label \
                            --cl-aug $k \
                            --aug-repeat-times $i \
                            --joint-da-loss $l \
                            --sub-max \
                            --cl_temp 0.07 \
                            --tlcl \
                            --ucl \
                            --skip-first-cl ucl+tlcl
                    done
                done
            done
        done
    done
done
