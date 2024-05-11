import torch

from fairseq.file_io import PathManager
from fairseq.data import Dictionary

LANG_GROUP_DICT = {
    "<unk>":"<unk>",
    "en":"germanic",
    "zu":"niger-congo",
    "bg":"slavic",
    "et":"uralic",
    "nb":"germanic",
    "ru":"slavic",
    "vi":"austroasiatic",
    "sk":"slavic",
    "tt":"turkic",
    "is":"germanic",
    "ko":"koreanic",
    "eo":"constructed",
    "sh":"slavic",
    "bs":"slavic",
    "mt":"afro-asiatic",
    "ka":"kartvelian",
    "th":"tai-kadai",
    "it":"romance",
    "cs":"slavic",
    "az":"turkic",
    "gu":"indo-iranian",
    "fy":"germanic",
    "ta":"dravidian",
    "ps":"indo-iranian",
    "rw":"niger-congo",
    "uz":"turkic",
    "ug":"turkic",
    "sq":"albanian",
    "fa":"indo-iranian",
    "an":"romance",
    "mk":"slavic",
    "nn":"germanic",
    "kk":"turkic",
    "tr":"turkic",
    "ky":"turkic",
    "wa":"romance",
    "ar":"afro-asiatic",
    "af":"germanic",
    "se":"uralic",
    "kn":"dravidian",
    "ha":"afro-asiatic",
    "lt":"baltic",
    "es":"romance",
    "lv":"baltic",
    "pl":"slavic",
    "ku":"indo-iranian",
    "sr":"slavic",
    "am":"afro-asiatic",
    "hr":"slavic",
    "ms":"austronesian",
    "or":"indo-iranian",
    "dz":"nilo-saharan",
    "be":"slavic",
    "my":"sino-tibetan",
    "li":"germanic",
    "tg":"indo-iranian",
    "ga":"celtic",
    "fi":"uralic",
    "nl":"germanic",
    "br":"celtic",
    "pt":"romance",
    "cy":"celtic",
    "oc":"romance",
    "mr":"indo-iranian",
    "km":"austroasiatic",
    "no":"germanic",
    "eu":"isolate",
    "gd":"celtic",
    "xh":"niger-congo",
    "mn":"mongolic",
    "de":"germanic",
    "he":"afro-asiatic",
    "fr":"romance",
    "id":"austronesian",
    "ur":"indo-iranian",
    "te":"dravidian",
    "sl":"slavic",
    "zh":"sino-tibetan",
    "hi":"indo-iranian",
    "ja":"japonic",
    "ig":"niger-congo",
    "ne":"indo-iranian",
    "yi":"germanic",
    "uk":"slavic",
    "mg":"austronesian",
    "hu":"uralic",
    "pa":"indo-iranian",
    "el":"hellenic",
    "yo":"niger-congo",
    "ca":"romance",
    "gl":"romance",
    "hy":"armenian",
    "bn":"indo-iranian",
    "ml":"dravidian",
    "tk":"turkic",
    "da":"germanic",
    "as":"indo-iranian",
    "sv":"germanic",
    "ro":"romance",
    "si":"dravidian",
}

def distance_of_two_tensor(t1, t2):
    return torch.max(torch.abs(t1-t2))

def is_inf(tensor):
    return torch.any(torch.isinf(tensor))

def is_nan(tensor):
    return torch.any(torch.isnan(tensor))

def inverse_sort(order):
    # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
    return torch.empty_like(order).scatter_(0, order, torch.arange(0, order.size(0), device=order.device))

def create_lang_dictionary(langs):
    unk = "<unk>"
    # hack to remove symbols other than unk as they are not needed by lang dict
    lang_dict = Dictionary(pad=unk, eos=unk, unk=unk, bos=unk)
    for lang in langs:
        lang_dict.add_symbol(lang)
    return lang_dict

def get_langtok_index(lang_tok, dic):
        idx = dic.index(lang_tok)
        assert (
            idx != dic.unk_index
        ), "cannot find language token {} in the dictionary".format(lang_tok)
        return idx
 
def convert_langdict_to_groupid(langdict_path: str):
    with open(
        PathManager.get_local_path(langdict_path), "r", encoding="utf-8"
    ) as f:
        langs = [lang.strip() for lang in f.readlines() if lang.strip()]
    
    lang_dict = create_lang_dictionary(langs)
    
    lang_groups = []
    for lang in lang_dict.symbols:
        if lang in LANG_GROUP_DICT:
            lang_groups.append(LANG_GROUP_DICT[lang])
        else:
            raise NotImplementedError
    
    lang_group_uniq = sorted(set(lang_groups))
    lang_group_index = {lang_group: i for i, lang_group in enumerate(lang_group_uniq)}
    lang_group_id = [lang_group_index[lang_group] for lang_group in lang_groups]
    
    return lang_group_id

if __name__ == "__main__":
    test_lang_dict = "test_lang_dict.txt"
    lang_group_id = convert_langdict_to_groupid(test_lang_dict)
    print(lang_group_id)    # should be [ 0, 10, 18, 22, 25, 10, 22,  4, 22, 24, 10, 16,  8, 22, 22,  1, 15, 23, 20, 22, 24, 12, 10,  9, 12, 18, 24, 24,  2, 12, 20, 22, 10, 24, 24, 24, 20,  1, 10, 25,  9,  1,  6, 20,  6, 22, 12, 22,  1, 22,  5, 12, 19, 22, 21, 10, 12,  7, 25, 10,  7, 20,  7, 20, 12,  4, 10, 13,  7, 18, 17, 10, 1, 20,  5, 12,  9, 22, 21, 12, 14, 18, 12, 10, 22,  5, 25, 12, 11, 18, 20, 20,  3, 12,  9, 24, 10, 12, 10, 20,  9]
                    