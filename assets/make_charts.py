# Solomon thread charts, 4 Nov 2026. neuralloot style: steel-blue paper, white cards,
# Gabarito titles, Hanken Grotesk body, IBM Plex Mono numbers. Every number is from
# README.md (ours, locked) or PUBLIC_COMPARES.md (public, cited). AUDIT=1 checks text edges/overlaps.
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as FP
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import os, sys
FD="/workspace/neuralloot/evening-2026-11-04/chart/fonts/"
OUT=os.path.dirname(os.path.abspath(__file__))
def F(n,s): return FP(fname=FD+n,size=s)
H8=lambda s:F("GB-800.ttf",s); H7=lambda s:F("GB-700.ttf",s)
TR=lambda s:F("HK-400.ttf",s); TB=lambda s:F("HK-600.ttf",s); TK=lambda s:F("HK-800.ttf",s)
MB=lambda s:F("PM-700.ttf",s); MS=lambda s:F("PM-600.ttf",s); MR=lambda s:F("PM-400.ttf",s)
BG="#e8edf4"; CARD="#ffffff"; EDGE="#c8d2e0"; INK="#141b2b"; SUB="#435069"; MUTE="#6f7a8f"
OURS="#0f9f74"; OURS2="#57c4a0"; PUB="#2453d6"; PUB2="#9fb0d6"; TAX="#e0533d"; AMB="#d99a0b"; TRACK="#eef2f7"; GRID="#e2e7ef"

class Canvas:
    def __init__(s,W,H):
        s.W,s.H=W,H; s.fig=plt.figure(figsize=(W/100,H/100),dpi=100); s.fig.patch.set_facecolor(BG)
        s.ax=s.fig.add_axes([0,0,1,1]); s.ax.set_xlim(0,W); s.ax.set_ylim(H,0); s.ax.axis("off"); s.TX=[]
    def T(s,x,y,t,fp,c=INK,**k):
        o=s.ax.text(x,y,t,fontproperties=fp,color=c,**k); s.TX.append(o); return o
    def card(s,x,y,w,h,fc=CARD,ec=EDGE,r=22):
        s.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=1.8,zorder=1))
    def save(s,name):
        p=os.path.join(OUT,name); s.fig.savefig(p,dpi=100,facecolor=BG)
        if os.environ.get("AUDIT"):
            rr=s.fig.canvas.get_renderer(); bad=0
            bbs=[(t,t.get_window_extent(rr)) for t in s.TX if t.get_text().strip()]
            for t,bb in bbs:
                if bb.x0<12 or bb.x1>s.W-12 or bb.y0<8 or bb.y1>s.H-8: print("EDGE",repr(t.get_text()[:40])); bad+=1
            for i in range(len(bbs)):
                for j in range(i+1,len(bbs)):
                    a,b=bbs[i][1],bbs[j][1]
                    if a.x0<b.x1-1 and b.x0<a.x1-1 and a.y0<b.y1-1 and b.y0<a.y1-1:
                        print("OVERLAP",repr(bbs[i][0].get_text()[:30]),repr(bbs[j][0].get_text()[:30])); bad+=1
            print(name,"audit issues:",bad)
        plt.close(s.fig); print(p)

def bars(c,rows,x0,x1,ytop,rh,vmax,ticks,labw):
    """rows: (label, sublabel, lo, hi_or_None, kind, valtext)"""
    X=lambda v:x0+v/vmax*(x1-x0)
    yb=ytop+len(rows)*rh
    for v in ticks:
        c.ax.plot([X(v),X(v)],[ytop-6,yb],color=GRID,lw=1.4,zorder=2)
        c.T(X(v),yb+6,f"{v}",MR(16),MUTE,ha="center",va="top")
    for i,(lab,sub,lo,hi,k,vt) in enumerate(rows):
        y=ytop+i*rh; bh=rh*0.46; by=y+rh*0.12
        col={"ours":OURS,"ours2":OURS2,"pub":PUB,"pub2":PUB2,"tax":TAX}[k]
        c.T(x0-labw,y+rh*0.26,lab,TK(18) if k.startswith("ours") or k=="tax" else TB(18),INK,va="center")
        c.T(x0-labw,y+rh*0.70,sub,TR(13),MUTE,va="center")
        c.ax.add_patch(Rectangle((x0,by),X(lo)-x0,bh,fc=col,ec="none",zorder=3))
        end=X(lo)
        if hi:
            c.ax.add_patch(Rectangle((X(lo),by),X(hi)-X(lo),bh,fc=col,alpha=0.38,ec="none",zorder=3)); end=X(hi)
        c.T(end+10,by+bh/2,vt,MB(21) if k.startswith("ours") else MS(19),col if k!="pub2" else "#5a6f9e",va="center")
    return yb

# ===================== 1. field compare, 1 request
c=Canvas(1600,1380)
L=56
c.T(L,40,"QWEN3.8-27B  ·  ONE RTX 5090  ·  1 REQUEST  ·  PUBLIC REPORTS",MB(18),PUB,va="top")
c.T(L,76,"Everyone's fast. Who's checked lossless under load?",H8(36),INK,va="top")
c.T(L,136,"Green = ours (wall clock, prefill counted, replies not cut, batch-invariant). Blue = published, each on its own meter.",TR(18),SUB,va="top")
c.card(36,180,1528,950)
rows=[
 ("Ours · thinking off","W4A16 · DFlash2 K=7 · 57k window · prefill in · invariant",290.5,None,"ours","290.5"),
 ("Ours · thinking on","same stack · reasoning + answer in one generate",228.7,None,"ours","228.7"),
 ("LukasParke · math","llama.cpp DFlash2 · IQ4_XS · ~131k · often decode-only",293,320,"pub","293–320"),
 ("LukasParke · prose","same fork · same meter",114,127,"pub","114–127"),
 ("gratex (HF card)","vLLM DFlash2 spec=7 · W4A16 g128 (our quant family) · fp8/bf16 KV",260,270,"pub","260–270"),
 ("darksidewalker","SGLang DFlash2 · NVFP4 · live-traffic median (peak 277)",221,None,"pub","221"),
 ("Gabe Ortiz / localLLMrig","q27 engine · 4-bit · Claude Code traffic · self-reported",220,None,"pub","~220"),
 ("GenerationAI (Medium)","llama.cpp DFlash2 (was ~146 with MTP)",215,None,"pub","215"),
 ("Suhail · SGLang DFlash2","NVFP4 · FP8 KV · matched 32k suite",139.7,None,"pub","139.7"),
 ("Suhail · baseline / MTP / DSpark","vLLM 84.5 · vLLM MTP 100.8 · SGLang DSpark 110.2",84.5,110.2,"pub2","84.5–110.2"),
 ("Ours · plain greedy, invariant","no drafter: the price of batch-invariant kernels",38,None,"tax","~38"),
]
yb=bars(c,rows,600,1420,206,78,350,[0,50,100,150,200,250,300,350],500)
c.T(600+820/2,yb+40,"tokens / second, one request",TB(17),MUTE,ha="center",va="top")
fy=1150
c.T(L,fy,"Not apples-to-apples. Meters: decode-only vs wall-with-prefill, live-traffic median vs fixed suite, self-reported vs scripted.",TR(16),SUB,va="top")
c.T(L,fy+30,"Quants: IQ4_XS GGUF, NVFP4, W4A16 GPTQ. Engines: llama.cpp, SGLang, vLLM, q27. Context: 32k–131k configs vs our 57,344.",TR(16),SUB,va="top")
c.T(L,fy+60,"\"Lossless\" elsewhere = greedy spec decode in principle, or pass/fail output gates. No public row states batch invariance.",TR(16),SUB,va="top")
c.T(L,fy+100,"Sources: suhailmohebi.com · github.com/darksidewalker · localllmrig.com · medium.com/@GenerationAI · hf.co/gratex · LukasParke fork",MR(12),INK,va="top")
c.T(1600-L,fy+140,"@neuralloot",MB(20),PUB,ha="right",va="top")
c.T(L,fy+142,"github.com/NeuralLoot/solomon-lossless-5090",MS(16),INK,va="top")
c.save("solomon-compare.png")

# ===================== 2. lossless + batch-invariant explainer
c=Canvas(1600,1000)
c.T(L,40,"WHAT \"LOSSLESS\" MEANS HERE",MB(18),OURS,va="top")
c.T(L,76,"Same tokens as plain greedy. Even 4 at a time.",H8(36),INK,va="top")
c.T(L,140,"Same 4-bit checkpoint, same prompt: speculative output = plain greedy output, token for token. Not a BF16 match (BF16 doesn't fit).",TR(17),SUB,va="top")
# panel A: speculative step
c.card(36,196,760,470)
c.T(66,220,"1 · Draft 7, verify 8, keep what matches",H7(22),INK,va="top")
c.T(66,260,"DFlash2 drafts a block. The 27B checks it in one forward pass.",TR(15),MUTE,va="top")
toks=["The","cat","sat","on","the","mat","and","purred"]
ok=[1,1,1,0,0,0,0]
bx=66; 
c.T(66,306,"drafter",TB(17),SUB,va="top")
for i,t in enumerate(toks[1:]):
    x=bx+90+i*90; c.ax.add_patch(FancyBboxPatch((x,300),80,40,boxstyle="round,pad=0,rounding_size=8",fc=TRACK,ec=EDGE,lw=1.2,zorder=2))
    c.T(x+40,320,t,MS(16),INK,ha="center",va="center")
c.T(66,386,"target",TB(17),SUB,va="top")
tgt=["cat","sat","on","the","rug",".","",""]
for i in range(7):
    x=bx+90+i*90
    good=ok[i]; fc="#dff5ec" if good else ("#fde4df" if i==3 else "#f3f4f6")
    ec=OURS if good else (TAX if i==3 else EDGE)
    c.ax.add_patch(FancyBboxPatch((x,380),80,40,boxstyle="round,pad=0,rounding_size=8",fc=fc,ec=ec,lw=1.6,zorder=2))
    c.T(x+40,400,"✓" if good else ("rug" if i==3 else "drop"),MB(16) if good else MS(14),ec if i<=3 else MUTE,ha="center",va="center")
c.T(66,450,"Matches kept. First miss → the target's own token (\"rug\").",TR(16),INK,va="top")
c.T(66,478,"Everything after it is dropped. Output = target argmax, always.",TR(16),INK,va="top")
c.ax.add_patch(FancyBboxPatch((66,526),700,112,boxstyle="round,pad=0,rounding_size=14",fc="#f5f7fb",ec=EDGE,lw=1.2,zorder=2))
c.T(90,546,"~5.67",MB(34),OURS,va="top"); c.T(270,552,"tokens kept per step, thinking off",TB(16),INK,va="top")
c.T(270,590,"~4.74 thinking on · one 27B forward per step",TR(14),MUTE,va="top")
# panel B: batch invariance
c.card(812,196,752,470)
c.T(842,220,"2 · A second user can't flip your token",H7(22),INK,va="top")
c.T(842,260,"Typical kernels change math order with batch size: tiny float",TR(15),MUTE,va="top")
c.T(842,290,"drift → different argmax → different reply. Ours are invariant.",TR(15),MUTE,va="top")
def lane(y,lab,others):
    c.T(842,y+20,lab,TB(18),SUB,va="center")
    for k in range(4):
        x=1010+k*126
        if k==0: fc,ec,t,tc=("#dff5ec",OURS,"you",OURS)
        elif k<=others: fc,ec,t,tc=(TRACK,EDGE,"req "+"BCD"[k-1],MUTE)
        else: fc,ec,t,tc=("#fafbfc","#e3e8ef","",MUTE)
        c.ax.add_patch(FancyBboxPatch((x,y),112,40,boxstyle="round,pad=0,rounding_size=8",fc=fc,ec=ec,lw=1.4,zorder=2))
        if t: c.T(x+56,y+20,t,MS(16),tc,ha="center",va="center")
lane(330,"alone",0); lane(392,"4 in flight",3)
c.T(1514,454,"→ identical tokens",TK(19),OURS,ha="right",va="top")
c.ax.add_patch(FancyBboxPatch((842,526),692,112,boxstyle="round,pad=0,rounding_size=14",fc="#f5f7fb",ec=EDGE,lw=1.2,zorder=2))
c.T(866,546,"16/16",MB(34),OURS,va="top"); c.T(1040,552,"prompts identical, incl. 4-wide",TB(16),INK,va="top")
c.T(1040,590,"last plain-vs-speculative identity check",TR(14),MUTE,va="top")
# bottom: the tax
c.card(36,686,1528,200)
c.T(66,708,"3 · The tax, and how the drafter pays it back",H7(22),INK,va="top")
X=lambda v:330+v/320*1150
for y,lab,v,col,vt in [(762,"plain, invariant",38,TAX,"~38 tok/s"),(818,"+ DFlash2 K=7",290.5,OURS,"290.5 tok/s")]:
    c.T(66,y+18,lab,TB(19),INK,va="center")
    c.ax.add_patch(Rectangle((330,y),1150,36,fc=TRACK,ec="none",zorder=2))
    c.ax.add_patch(Rectangle((330,y),X(v)-330,36,fc=col,ec="none",zorder=3))
    if v<100: c.T(X(v)+12,y+18,vt,MB(19),col,va="center")
    else: c.T(X(v)-12,y+18,vt,MB(19),"#ffffff",ha="right",va="center")
c.T(L,912,"Invariant kernels make plain greedy slow (~38 tok/s). Speculation buys it back without changing a token (1 request, thinking off, prefill counted).",TR(15),SUB,va="top")
c.T(L,950,"github.com/NeuralLoot/solomon-lossless-5090",MS(16),INK,va="top")
c.T(1600-L,948,"@neuralloot",MB(20),PUB,ha="right",va="top")
c.save("solomon-lossless.png")

# ===================== 3. under load: ours 1 vs 4 + public multi-request
c=Canvas(1600,1000)
c.T(L,40,"UNDER LOAD  ·  AGGREGATE TOK/S, ONE RTX 5090",MB(18),OURS,va="top")
c.T(L,76,"Four users. Same answers each would get alone.",H8(36),INK,va="top")
c.T(L,140,"Ours: 16 prompts, temp 0, wall clock with prefill, replies not cut, window 57,344. Public rows: their own meters.",TR(17),SUB,va="top")
c.card(36,196,1528,610)
rows=[
 ("Ours · thinking off · 4 at once","batch-invariant · 1 request alone: 290.5",614.3,None,"ours","614.3"),
 ("Ours · thinking on · 4 at once","batch-invariant · 1 request alone: 228.7",467.4,None,"ours","467.4"),
 ("LukasParke · 4 streams","llama.cpp DFlash2 · IQ4_XS · ~131k · decode meter",285,None,"pub","~285"),
 ("gratex · 8 at once, spec=7","vLLM DFlash2 · W4A16 · fp8 / bf16 KV",776,868,"pub2","776–868"),
 ("gratex · 8 at once, spec=3","same card · no invariance stated",995,1013,"pub2","995–1,013"),
]
yb=bars(c,rows,560,1420,232,100,1100,[0,200,400,600,800,1000],500)
c.T(560+430,yb+34,"tokens / second, all requests combined",TB(17),MUTE,ha="center",va="top")
c.T(L,830,"gratex runs 8 requests (we run 4) and states no token-identity check; more concurrency buys raw throughput.",TR(16),SUB,va="top")
c.T(L,860,"We trade some of that for a guarantee: request #2 can't change a token request #1 sees.",TR(16),SUB,va="top")
c.T(L,914,"Sources: hf.co/gratex/Qwen3.8-27B-DFlash2-W4A16-g128-sym-GPTQ · LukasParke llama.cpp fork",MR(12),INK,va="top")
c.T(1600-L,948,"@neuralloot",MB(20),PUB,ha="right",va="top")
c.save("solomon-load.png")
