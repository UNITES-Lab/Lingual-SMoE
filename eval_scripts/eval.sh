enc_langtok=src
subset=test

function init(){
    echo save_dir=$save_dir
    echo subset=$subset

    model_name=(${save_dir//// })
    model_name=${model_name[-1]}
    model_name=${model_name}

    echo "model_name:${model_name}"
    echo "prefix:${prefix}"
    echo "translation_dir:${translation_dir}"
    echo "score_path:${score_path}"

    python=python
    sacrebleu=sacrebleu

    translation_dir=translation_data/$model_name
    score_path=bleu/$model_name.bleu
}

function eval(){
    if [ $data_type == 'opus16' ]; then
        all_lang_pairs="en-eu,en-pt,en-bg,en-sk,en-zh,en-sl,en-de,en-hr,en-nb,en-ga,en-rw,en-as,en-fy,en-mr,en-se,eu-en,pt-en,bg-en,sk-en,zh-en,sl-en,de-en,hr-en,nb-en,ga-en,rw-en,as-en,fy-en,mr-en,se-en"
    elif [ $data_type == 'opus50' ]; then
        all_lang_pairs="en-eu,en-pt,en-bg,en-sk,en-zh,en-sl,en-de,en-hr,en-nb,en-ga,en-rw,en-as,en-fy,en-mr,en-se,eu-en,pt-en,bg-en,sk-en,zh-en,sl-en,de-en,hr-en,nb-en,ga-en,rw-en,as-en,fy-en,mr-en,se-en,hu-en,az-en,uz-en,ug-en,ky-en,ig-en,zu-en,ko-en,ja-en,pl-en,uk-en,sh-en,ro-en,es-en,gl-en,fa-en,bn-en,gu-en,tg-en,ps-en,ku-en,el-en,da-en,nl-en,af-en,nn-en,yi-en,gd-en,lt-en,lv-en,sq-en,ml-en,ha-en,en-hu,en-az,en-uz,en-ug,en-ky,en-ig,en-zu,en-ko,en-ja,en-pl,en-uk,en-sh,en-ro,en-es,en-gl,en-fa,en-bn,en-gu,en-tg,en-ps,en-ku,en-el,en-da,en-nl,en-af,en-nn,en-yi,en-gd,en-lt,en-lv,en-sq,en-ml,en-ha"
    elif [ $data_type == 'opus100' ]; then
        all_lang_pairs="en-fr,cy-en,hu-en,en-lt,en-mg,yi-en,as-en,en-mr,uz-en,eo-en,li-en,es-en,ka-en,am-en,en-he,en-ja,nb-en,en-ku,en-cs,en-fi,si-en,en-no,en-se,az-en,en-ga,da-en,en-vi,eu-en,en-pa,ca-en,id-en,en-eu,cs-en,kn-en,te-en,en-ug,en-be,rw-en,gu-en,en-cy,en-tt,en-am,xh-en,en-nb,sv-en,sq-en,en-nn,en-bn,ha-en,en-hu,en-pl,en-ko,en-tg,en-zu,en-nl,ps-en,af-en,be-en,ga-en,mg-en,en-mt,bs-en,or-en,bn-en,en-sr,tg-en,hi-en,fr-en,se-en,en-hr,en-eo,en-de,en-it,sk-en,tt-en,is-en,km-en,en-br,nn-en,vi-en,en-ka,ne-en,en-et,ro-en,en-ha,fa-en,oc-en,en-sh,ko-en,en-yi,en-fa,it-en,no-en,en-ig,en-af,en-da,en-th,ur-en,en-pt,zu-en,ja-en,zh-en,ar-en,en-ky,fi-en,en-mk,lv-en,my-en,en-kk,ta-en,en-ca,mt-en,fy-en,en-uk,th-en,el-en,ml-en,et-en,en-my,en-es,en-sv,wa-en,en-sk,en-ro,en-oc,bg-en,en-uz,tr-en,sl-en,sh-en,de-en,en-lv,en-is,en-km,mr-en,en-hi,pa-en,en-gu,hr-en,en-tk,en-ta,pl-en,en-kn,lt-en,en-ps,ug-en,en-bg,br-en,en-ru,en-sl,en-ne,en-te,en-bs,tk-en,gl-en,en-si,en-rw,sr-en,pt-en,en-tr,ky-en,en-gd,ku-en,en-id,en-ur,en-li,uk-en,en-or,en-sq,gd-en,en-ar,en-ml,kk-en,en-el,en-zh,en-gl,en-as,ig-en,ms-en,nl-en,en-fy,en-az,he-en,en-ms,ru-en,mk-en,en-wa,en-xh"
    else
        echo "unknown data type $data_type"
        exit 1
    fi

    lang_pairs=${all_lang_pairs//,/ }

    for lang_pair in ${lang_pairs// / }; do
        array=(${lang_pair//-/ })
        src_lang=${array[0]}
        tgt_lang=${array[1]}

        parallel_trans_dir=${translation_dir}/${lang_pair}
        echo "compute bleu for ${lang_pair}"
        ${python} -u ./translation_utils/extract_translation.py \
            --translation_file_path ${parallel_trans_dir}/generate-${subset}.txt \
            --output_hp_file_path ${parallel_trans_dir}/extract.${subset}.txt \
            --output_ref_file_path ${parallel_trans_dir}/gt.${subset}.txt \
        
        score=$(${sacrebleu} -l ${lang_pair} -w 6 ${parallel_trans_dir}/gt.${subset}.txt < ${parallel_trans_dir}/extract.${subset}.txt)
        
        score=$(echo $score | grep -Po ":\s(\d+\.*\d*)" | head -n 1 | grep -Po "\d+\.*\d*")
        
        echo "${lang_pair}: ${score}" >> ${score_path}
    done

    echo "compute average subset bleu score"
    ${python} ./translation_utils/average_subset_bleu.py ${score_path}
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
        --subset)
            subset=$2
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

eval