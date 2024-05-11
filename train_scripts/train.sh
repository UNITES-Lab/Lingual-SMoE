export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=""
export WANDB_API_KEY=""

task_name=test

function opus16_data_init(){
    data_dir=/path/to/opus-16-preprocessed
    data_args=" ${data_dir}/main_data_bin \
            --lang-dict ${data_dir}/lang_dict.txt \
            --lang-pairs en-eu,en-pt,en-bg,en-sk,en-zh,en-sl,en-de,en-hr,en-nb,en-ga,en-rw,en-as,en-fy,en-mr,en-se,eu-en,pt-en,bg-en,sk-en,zh-en,sl-en,de-en,hr-en,nb-en,ga-en,rw-en,as-en,fy-en,mr-en,se-en \
            --encoder-langtok src \
            --decoder-langtok \
            --enable-lang-ids \
            --sampling-method temperature \
            --sampling-temperature 1.5 \
            --source-dict ${data_dir}/main_data_bin/dict.txt \
            --target-dict ${data_dir}/main_data_bin/dict.txt \
            --task translation_multi_simple_epoch \
            --num-workers 8"          
}

function opus50_data_init(){
    data_dir=/path/to/opus-50-preprocessed
    data_args=" ${data_dir}/main_data_bin \
            --lang-dict ${data_dir}/lang_dict.txt \
            --lang-pairs en-eu,en-pt,en-bg,en-sk,en-zh,en-sl,en-de,en-hr,en-nb,en-ga,en-rw,en-as,en-fy,en-mr,en-se,eu-en,pt-en,bg-en,sk-en,zh-en,sl-en,de-en,hr-en,nb-en,ga-en,rw-en,as-en,fy-en,mr-en,se-en,hu-en,az-en,uz-en,ug-en,ky-en,ig-en,zu-en,ko-en,ja-en,pl-en,uk-en,sh-en,ro-en,es-en,gl-en,fa-en,bn-en,gu-en,tg-en,ps-en,ku-en,el-en,da-en,nl-en,af-en,nn-en,yi-en,gd-en,lt-en,lv-en,sq-en,ml-en,ha-en,en-hu,en-az,en-uz,en-ug,en-ky,en-ig,en-zu,en-ko,en-ja,en-pl,en-uk,en-sh,en-ro,en-es,en-gl,en-fa,en-bn,en-gu,en-tg,en-ps,en-ku,en-el,en-da,en-nl,en-af,en-nn,en-yi,en-gd,en-lt,en-lv,en-sq,en-ml,en-ha \
            --encoder-langtok src \
            --decoder-langtok \
            --enable-lang-ids \
            --sampling-method temperature \
            --sampling-temperature 5 \
            --source-dict ${data_dir}/main_data_bin/dict.txt \
            --target-dict ${data_dir}/main_data_bin/dict.txt \
            --task translation_multi_simple_epoch \
            --num-workers 8"     
}

function opus100_data_init(){
    data_dir=/path/to/opus-100-preprocessed
    data_args=" ${data_dir}/main_data_bin \
            --lang-dict ${data_dir}/lang_dict.txt \
            --lang-pairs en-fr,cy-en,hu-en,en-lt,en-mg,yi-en,as-en,en-mr,uz-en,eo-en,li-en,es-en,ka-en,am-en,en-he,en-ja,nb-en,en-ku,en-cs,en-fi,si-en,en-no,en-se,az-en,en-ga,da-en,en-vi,eu-en,en-pa,ca-en,id-en,en-eu,cs-en,kn-en,te-en,en-ug,en-be,rw-en,gu-en,en-cy,en-tt,en-am,xh-en,en-nb,sv-en,sq-en,en-nn,en-bn,ha-en,en-hu,en-pl,en-ko,en-tg,en-zu,en-nl,ps-en,af-en,be-en,ga-en,mg-en,en-mt,bs-en,or-en,bn-en,en-sr,tg-en,hi-en,fr-en,se-en,en-hr,en-eo,en-de,en-it,sk-en,tt-en,is-en,km-en,en-br,nn-en,vi-en,en-ka,ne-en,en-et,ro-en,en-ha,fa-en,oc-en,en-sh,ko-en,en-yi,en-fa,it-en,no-en,en-ig,en-af,en-da,en-th,ur-en,en-pt,zu-en,ja-en,zh-en,ar-en,en-ky,fi-en,en-mk,lv-en,my-en,en-kk,ta-en,en-ca,mt-en,fy-en,en-uk,th-en,el-en,ml-en,et-en,en-my,en-es,en-sv,wa-en,en-sk,en-ro,en-oc,bg-en,en-uz,tr-en,sl-en,sh-en,de-en,en-lv,en-is,en-km,mr-en,en-hi,pa-en,en-gu,hr-en,en-tk,en-ta,pl-en,en-kn,lt-en,en-ps,ug-en,en-bg,br-en,en-ru,en-sl,en-ne,en-te,en-bs,tk-en,gl-en,en-si,en-rw,sr-en,pt-en,en-tr,ky-en,en-gd,ku-en,en-id,en-ur,en-li,uk-en,en-or,en-sq,gd-en,en-ar,en-ml,kk-en,en-el,en-zh,en-gl,en-as,ig-en,ms-en,nl-en,en-fy,en-az,he-en,en-ms,ru-en,mk-en,en-wa,en-xh \
            --encoder-langtok src \
            --decoder-langtok \
            --enable-lang-ids \
            --sampling-method temperature \
            --sampling-temperature 5 \
            --source-dict ${data_dir}/main_data_bin/dict.txt \
            --target-dict ${data_dir}/main_data_bin/dict.txt \
            --task translation_multi_simple_epoch \
            --num-workers 8"           
}

function test_data_init(){
    data_dir=/path/to/opus-16-preprocessed
    data_args=" ${data_dir}/main_data_bin \
            --lang-dict ${data_dir}/lang_dict.txt \
            --lang-pairs en-fy,en-mr,en-se,fy-en,mr-en,se-en \
            --encoder-langtok src \
            --decoder-langtok \
            --enable-lang-ids \
            --sampling-method temperature \
            --sampling-temperature 1.5 \
            --source-dict ${data_dir}/main_data_bin/dict.txt \
            --target-dict ${data_dir}/main_data_bin/dict.txt \
            --task translation_multi_simple_epoch \
            --num-workers 8"          
}

function global_setting_init(){
    ddp=fully_sharded

    save_args=" --save-dir output/${task_name} \
            --validate-interval-updates 100 \
            --save-interval-updates 100 \
            --keep-interval-updates 1 \
            --no-epoch-checkpoints \
            --no-save-optimizer-state-on-training-finished"
    moe_args="--arch smoe \
            --wandb-project moe_mmt \
            --moe-gating-use-fp32 \
            --moe-second-expert-policy all \
            --moe-normalize-expert-grad sqrt_world_size \
            --criterion moe_cross_entropy \
            --moe-gate-loss-wt 0.05 \
            --moe-gate-loss-combine-method sum \
            --moe-batch-prioritized-routing \
            --use-moe-pad-mask \
            --moe-freq 2 \
            --moe-expert-count 32 \
            --add-lang-loss \
            --hmoe-gate \
            --task-mlp-path assets/task_mlp_weight.pt"
}

function train_single_node(){
    python train.py \
    $data_args \
    --max-tokens 8192 \
    --share-all-embeddings \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 1.0 \
    --lr 0.0005 \
    --warmup-updates 4000 \
    --lr-scheduler inverse_sqrt \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --num-workers-valid 0 \
    --max-update 35000 \
    --ddp-backend ${ddp} \
    --user-dir ./smoe \
    --best-checkpoint-metric ppl \
    --log-format simple \
    --log-interval 100 \
    --fp16 \
    $save_args \
    $model_args \
    $moe_args
}

test_data_init
# opus50_data_init
# opus100_data_init
# opus16_data_init

global_setting_init

train_single_node 
