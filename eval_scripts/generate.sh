export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=""

best_or_last=best
enc_langtok=src
subset=test
capacity_factor=0.5
n_process=8

function init(){
    echo save_dir=$save_dir
    echo subset=$subset
    echo best_or_last=$best_or_last

    model_name=(${save_dir//// })
    model_name=${model_name[-1]}
    model_name=${model_name}

    echo "model_name:${model_name}"
    echo "translation_dir:${translation_dir}"

    master_addr="127.0.0.3"
    master_port=12345

    max_tokens=6000

    python=python
    sacrebleu=sacrebleu

    checkpoint="checkpoint_${best_or_last}"
    checkpoint_path="${save_dir}/${checkpoint}.pt"

    main_data_bin_dir=${root_data_dir}/main_data_bin
    extra_data_bin_dir=${root_data_dir}/extra_data_bin
    spm_data_dir=${root_data_dir}/spm_data
    spm_corpus_dir=${root_data_dir}/spm_corpus
    lang_dict=${root_data_dir}/lang_dict.txt
    
    translation_dir=translation_data/$model_name
    
    mkdir -p ${translation_dir}
    result_path=${translation_dir}
    echo "write translation to ${translation_dir}"
}


function init_data(){
    if [ $data_type == 'opus16' ]; then
        root_data_dir=/path/to/opus-16-preprocessed
        all_lang_pairs="en-eu,en-pt,en-bg,en-sk,en-zh,en-sl,en-de,en-hr,en-nb,en-ga,en-rw,en-as,en-fy,en-mr,en-se,eu-en,pt-en,bg-en,sk-en,zh-en,sl-en,de-en,hr-en,nb-en,ga-en,rw-en,as-en,fy-en,mr-en,se-en"
    elif [ $data_type == 'opus50' ]; then
        root_data_dir=/path/to/opus-50-preprocessed
        all_lang_pairs="en-eu,en-pt,en-bg,en-sk,en-zh,en-sl,en-de,en-hr,en-nb,en-ga,en-rw,en-as,en-fy,en-mr,en-se,eu-en,pt-en,bg-en,sk-en,zh-en,sl-en,de-en,hr-en,nb-en,ga-en,rw-en,as-en,fy-en,mr-en,se-en,hu-en,az-en,uz-en,ug-en,ky-en,ig-en,zu-en,ko-en,ja-en,pl-en,uk-en,sh-en,ro-en,es-en,gl-en,fa-en,bn-en,gu-en,tg-en,ps-en,ku-en,el-en,da-en,nl-en,af-en,nn-en,yi-en,gd-en,lt-en,lv-en,sq-en,ml-en,ha-en,en-hu,en-az,en-uz,en-ug,en-ky,en-ig,en-zu,en-ko,en-ja,en-pl,en-uk,en-sh,en-ro,en-es,en-gl,en-fa,en-bn,en-gu,en-tg,en-ps,en-ku,en-el,en-da,en-nl,en-af,en-nn,en-yi,en-gd,en-lt,en-lv,en-sq,en-ml,en-ha"
    elif [ $data_type == 'opus100' ]; then
        root_data_dir=/path/to/opus-100-preprocessed
        all_lang_pairs="en-fr,cy-en,hu-en,en-lt,en-mg,yi-en,as-en,en-mr,uz-en,eo-en,li-en,es-en,ka-en,am-en,en-he,en-ja,nb-en,en-ku,en-cs,en-fi,si-en,en-no,en-se,az-en,en-ga,da-en,en-vi,eu-en,en-pa,ca-en,id-en,en-eu,cs-en,kn-en,te-en,en-ug,en-be,rw-en,gu-en,en-cy,en-tt,en-am,xh-en,en-nb,sv-en,sq-en,en-nn,en-bn,ha-en,en-hu,en-pl,en-ko,en-tg,en-zu,en-nl,ps-en,af-en,be-en,ga-en,mg-en,en-mt,bs-en,or-en,bn-en,en-sr,tg-en,hi-en,fr-en,se-en,en-hr,en-eo,en-de,en-it,sk-en,tt-en,is-en,km-en,en-br,nn-en,vi-en,en-ka,ne-en,en-et,ro-en,en-ha,fa-en,oc-en,en-sh,ko-en,en-yi,en-fa,it-en,no-en,en-ig,en-af,en-da,en-th,ur-en,en-pt,zu-en,ja-en,zh-en,ar-en,en-ky,fi-en,en-mk,lv-en,my-en,en-kk,ta-en,en-ca,mt-en,fy-en,en-uk,th-en,el-en,ml-en,et-en,en-my,en-es,en-sv,wa-en,en-sk,en-ro,en-oc,bg-en,en-uz,tr-en,sl-en,sh-en,de-en,en-lv,en-is,en-km,mr-en,en-hi,pa-en,en-gu,hr-en,en-tk,en-ta,pl-en,en-kn,lt-en,en-ps,ug-en,en-bg,br-en,en-ru,en-sl,en-ne,en-te,en-bs,tk-en,gl-en,en-si,en-rw,sr-en,pt-en,en-tr,ky-en,en-gd,ku-en,en-id,en-ur,en-li,uk-en,en-or,en-sq,gd-en,en-ar,en-ml,kk-en,en-el,en-zh,en-gl,en-as,ig-en,ms-en,nl-en,en-fy,en-az,he-en,en-ms,ru-en,mk-en,en-wa,en-xh"
    else
        echo "unknown data type $data_type"
        exit 1
    fi

    lang_pairs=${all_lang_pairs//,/ }
}

function generate(){
    # for generate_multiple.py, --source-lang and --target-lang does not work, it would iterate all languages in lang-pairs-to-generate
    ${python} generate_multiple.py ${main_data_bin_dir} \
    --task translation_multi_simple_epoch \
    --user-dir ./smoe \
    --distributed-world-size ${n_process} \
    --lang-pairs ${all_lang_pairs} \
    --lang-dict ${lang_dict} \
    --source-dict ${main_data_bin_dir}/dict.txt \
    --target-dict ${main_data_bin_dir}/dict.txt \
    --decoder-langtok \
    --encoder-langtok src \
    --enable-lang-ids \
    --source-lang en \
    --target-lang eu \
    --gen-subset ${subset} \
    --path ${checkpoint_path} \
    --max-tokens ${max_tokens} \
    --beam 5 \
    --results-path ${result_path} \
    --post-process sentencepiece \
    --lang-pairs-to-generate $lang_pairs \
    --skip-invalid-size-inputs-valid-test \
    --model-overrides "{'moe_eval_capacity_token_fraction':${capacity_factor}}" \
    --ddp-backend fully_sharded \
    --is-moe
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data_type)
            data_type=$2
            shift
            shift
            ;;
        -s|--save_dir)
            save_dir=$2
            shift
            shift
            ;;
        -n|--n_process)
            n_process=$2
            shift
            shift
            ;;
        --subset)
            subset=$2
            shift
            shift
            ;;
        --last)
            best_or_last=last
            shift
            ;;
        -c|--capacity_factor )
            capacity_factor=$2
            shift
            shift
            ;;
        -*|--*)
            echo "unkown option $1"
            exit 1
            ;;
    esac
done

init

init_data

generate
