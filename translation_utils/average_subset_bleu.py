import sys

if len(sys.argv)>1:
    fpath=sys.argv[1]
    with open(fpath, 'r') as fi:
        data=fi.read()
else:
    data=sys.stdin.read()

lang_bleu=dict()

high_lang=["zh","eu","bg","sk","sl","hr","pt","de","hu","ko","ja","pl","uk","ro","es","fa","bn","el","da","nl","lt","lv","sq","et","fi","tr","th","ru","bs","sr","cs","mk","fr","it","ca","sv","is","no","si","ms","id","vi","he","ar"]
mid_lang=["rw","as","nb","ga","az","uz","sh","gl","gu","tg","af","nn","ml","xh","ka","wa","ne","pa","ur","hi","cy","br","ta","eo","mg","km","mt"]
low_lang=["se","mr","fy","ug","ky","ig","zu","ps","ku","yi","gd","ha","kk","tt","tk","my","be","oc","or","li","te","kn","am"]

high_lang_bleu=dict()
mid_lang_bleu=dict()
low_lang_bleu=dict()
for l in data.split('\n'):
    if not (len(l.strip())>0 and len(l.split())>1):
        continue
    lang, bleu=l.split()[0], float(l.split()[1])
    # if lang in lang_bleu and lang_bleu[lang]!=bleu:
        # print(f'Warning: mulitple BLEU scores of language {lang}, overfit with the newest score')
    lang_bleu[lang]=bleu#+0.25

    src, tgt = lang.strip(':').split('-')

    if src in high_lang or tgt in high_lang:
        high_lang_bleu[lang]=bleu
    if src in mid_lang or tgt in mid_lang:
        mid_lang_bleu[lang]=bleu
    if src in low_lang or tgt in low_lang:
        low_lang_bleu[lang]=bleu

all_bleu=lang_bleu.values()
en_any=[b for l,b in lang_bleu.items() if l.startswith('en')]
any_en=[b for l,b in lang_bleu.items() if l.endswith('en:')]

high_all_bleu=high_lang_bleu.values()
high_en_any=[b for l,b in high_lang_bleu.items() if l.startswith('en')]
high_any_en=[b for l,b in high_lang_bleu.items() if l.endswith('en:')]

mid_all_bleu=mid_lang_bleu.values()
mid_en_any=[b for l,b in mid_lang_bleu.items() if l.startswith('en')]
mid_any_en=[b for l,b in mid_lang_bleu.items() if l.endswith('en:')]

low_all_bleu=low_lang_bleu.values()
low_en_any=[b for l,b in low_lang_bleu.items() if l.startswith('en')]
low_any_en=[b for l,b in low_lang_bleu.items() if l.endswith('en:')]

def avg_of_list(l):
    return sum(l)/len(l)

print('avg bleu:-------------')
print('language number: ', len(all_bleu))
print('all:', avg_of_list(all_bleu))
print('en-any:', avg_of_list(en_any))
print('any-en:', avg_of_list(any_en))

print('high language number: ', len(high_all_bleu))
print('high all:', avg_of_list(high_all_bleu))
print('high en-any:', avg_of_list(high_en_any))
print('high any-en:', avg_of_list(high_any_en))
print('mid language number: ', len(mid_all_bleu))
print('mid all:', avg_of_list(mid_all_bleu))
print('mid en-any:', avg_of_list(mid_en_any))
print('mid any-en:', avg_of_list(mid_any_en))
print('low language number: ', len(low_all_bleu))
print('low all:', avg_of_list(low_all_bleu))
print('low en-any:', avg_of_list(low_en_any))
print('low any-en:', avg_of_list(low_any_en))
