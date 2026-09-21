from PIL import Image, ImageDraw, ImageFont

W, H = 2400, 1500
img = Image.new('RGB', (W, H), 'white')
d = ImageDraw.Draw(img)

def font(size, bold=False):
    candidates = [
        'C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/calibrib.ttf' if bold else 'C:/Windows/Fonts/calibri.ttf'
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()

title_f = font(42, True)
head_f = font(30, True)
body_f = font(24)
small_f = font(21)

blue = '#174A7E'; teal = '#147D87'; gold = '#C58A13'; red = '#A9433D'; gray = '#EAF0F5'; dark = '#193247'

def rounded(box, fill, outline=dark, r=24, width=4):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)

def centered(text, box, f, fill=dark, spacing=4):
    x1,y1,x2,y2 = box
    lines = text.split('\n')
    heights = [d.textbbox((0,0), line, font=f)[3] for line in lines]
    total = sum(heights) + spacing*(len(lines)-1)
    y = (y1+y2-total)//2
    for line,h in zip(lines, heights):
        bb=d.textbbox((0,0), line, font=f)
        x=(x1+x2-(bb[2]-bb[0]))//2
        d.text((x,y), line, font=f, fill=fill)
        y += h+spacing

def arrow(x1,y1,x2,y2, color=dark, width=7):
    d.line((x1,y1,x2,y2), fill=color, width=width)
    import math
    ang=math.atan2(y2-y1,x2-x1)
    L=24
    p1=(x2-L*math.cos(ang-0.5), y2-L*math.sin(ang-0.5))
    p2=(x2-L*math.cos(ang+0.5), y2-L*math.sin(ang+0.5))
    d.polygon([(x2,y2),p1,p2], fill=color)

d.text((80,40), 'Explainable Bangla Fake Review Detection Pipeline', font=title_f, fill=dark)
d.text((80,100), 'Final-run methodology: audit → leakage control → feature fusion → controlled pseudo-labeling → locked evaluation', font=small_f, fill=blue)

# Inputs
rounded((80,180,500,340), '#DCEAF7', blue)
centered('Gold BFRD\n9,049 raw reviews', (90,190,490,330), head_f, blue)
rounded((80,390,500,550), '#E1F2EE', teal)
centered('BanglishRev\n1,746,943 scanned', (90,400,490,540), head_f, teal)
arrow(500,260,650,360); arrow(500,470,650,360)

rounded((650,275,1050,445), '#F2F5F7', dark)
centered('Schema audit + cleaning\nnormalization, redaction,\ninvalid-row removal', (660,285,1040,435), head_f)

arrow(1050,360,1200,360)
rounded((1200,275,1600,445), '#FFF1D6', gold)
centered('Duplicate control\nexact hashes + MinHash LSH\n3-token shingles, Jaccard', (1210,285,1590,435), head_f, '#72510A')

arrow(1400,445,1400,570)
rounded((1120,570,1680,720), '#E9E4F7', '#68529A')
centered('Duplicate-aware stratified split\nTrain 6,321 | Validation 1,356\nLocked test 1,356', (1130,580,1670,710), head_f, '#4D3979')

# feature branches
arrow(1200,720,520,875); arrow(1400,720,1000,875); arrow(1600,720,1500,875)
rounded((180,875,820,1110), '#DCEAF7', blue)
centered('Text representations\nTF–IDF: 30,000 terms\nWord2Vec: 25 dimensions\nBangla/Banglish tokenization', (190,885,810,1100), head_f, blue)
rounded((850,875,1370,1110), '#E1F2EE', teal)
centered('Handcrafted signals\nlength, code-mix, script counts\nURLs, prices, emojis, promotion\n26 behavioral features', (860,885,1360,1100), head_f, teal)
rounded((1400,875,2140,1110), '#FFF1D6', '#A36B00')
centered('Duplicate graph signals\ncluster size, duplicate flag\nnear-duplicate degree, similarity\ncluster size', (1410,885,2130,1100), head_f, '#72510A')

arrow(820,1110,1110,1200); arrow(1110,1110,1110,1200); arrow(1510,1110,1110,1200)
rounded((760,1190,1460,1360), '#E8F0FA', blue)
centered('Model-ready feature fusion\nLinear / NB: 30,026 dimensions\nTree: 51 dimensions\nclass-weighted training', (770,1200,1450,1350), head_f, blue)

arrow(1460,1275,1680,1275)
rounded((1680,1180,2310,1370), '#F5E2E2', red)
centered('Validation selection + explanations\nRandom Forest selected\nconfidence-capped pseudo-labels\nmetrics, LIME, error analysis', (1690,1190,2300,1360), head_f, red)

# small pseudo-label loop annotation
d.rounded_rectangle((1670,480,2310,720), radius=24, fill='#F7F7F7', outline=red, width=4)
centered('Pseudo-label branch\nLogistic Regression generator\nP(fake)>0.90, P(genuine)>0.75\nclass caps: fake 25,284; genuine 6,321\nweight = 0.50; max ratio = 5:1', (1680,490,2300,710), small_f, red)
arrow(1600,445,1900,480, red, 5)
arrow(1900,720,1460,1190, red, 5)

img.save('report_assets/pipeline_methodology.png', quality=95)
