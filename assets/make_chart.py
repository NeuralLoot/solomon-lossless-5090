import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
import glob
for f in glob.glob("/usr/share/fonts/truetype/sand-box/custom/Open Sauce Sans/OpenSauceSans-[RB]*.ttf"): fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "Open Sauce Sans"

BG="#0b0f14"; FG="#e8edf2"; DIM="#8a96a3"
OURS="#19e3a1"; OLD="#2f7d66"; PUB="#6b7785"; PLAIN="#3a4450"

# (label, low, high, kind, note)  kind: ours/old/pub/plain ; hatch if decode-only meter
single = [
 ("Ours · thinking off · 4 Nov",            290.5, None, "ours", "prefill in · not cut · 57k"),
 ("Ours · thinking on · one generate",      228.7, None, "ours", "prefill in · not cut · 57k"),
 ("Ours · earlier boot, off (older)",       253.3, None, "old",  "prefill in"),
 ("Ours · thinking on, stop+prefill (older)",180.3, None, "old", "prefill in"),
 ("Ours · plain greedy, batch-invariant",    38.0, None, "plain","16 Oct · the cost of invariance"),
 ("LukasParke llama.cpp · DFlash2 · math",  293,  320,  "pub",  "IQ4_XS · ~131k · often decode-only"),
 ("LukasParke llama.cpp · DFlash2 · prose", 114,  127,  "pub",  "IQ4_XS · ~131k · often decode-only"),
 ("Published plain decode, this family",    115,  None, "pub",  "decode-only"),
]
multi = [
 ("Ours · thinking off · 4 Nov",            614.3, None, "ours", "prefill in · not cut · 57k"),
 ("Ours · thinking on · one generate",      467.4, None, "ours", "prefill in · not cut · 57k"),
 ("Ours · earlier boot, off (older)",       537.8, None, "old",  "prefill in"),
 ("Ours · DFlash 16 Oct (older, 32k)",      440,   None, "old",  "16/16 identity at 4-wide"),
 ("Ours · thinking on, stop+prefill (older)",352.1, None, "old", "prefill in"),
 ("Ours · plain greedy, batch-invariant",   132.8, None, "plain","16 Oct"),
 ("LukasParke llama.cpp · DFlash2 · 4 streams",285, None, "pub", "IQ4_XS · ~131k · decode-only meter"),
]
col={"ours":OURS,"old":OLD,"pub":PUB,"plain":PLAIN}

fig,axes=plt.subplots(1,2,figsize=(16,9),dpi=100,facecolor=BG)
fig.subplots_adjust(left=0.235,right=0.975,top=0.80,bottom=0.17,wspace=0.95)
def panel(ax,rows,title,xmax):
    ax.set_facecolor(BG)
    n=len(rows)
    for i,(lab,lo,hi,k,note) in enumerate(rows):
        y=n-1-i
        hatch="///" if k=="pub" else None
        if hi:
            ax.barh(y,lo,color=col[k],height=0.62,hatch=hatch,edgecolor=BG,lw=0)
            ax.barh(y,hi-lo,left=lo,color=col[k],alpha=0.45,height=0.62,hatch=hatch,edgecolor=BG,lw=0)
            val=f"{lo:g}–{hi:g}"; end=hi
        else:
            ax.barh(y,lo,color=col[k],height=0.62,hatch=hatch,edgecolor=BG,lw=0)
            val=(f"~{lo:g}" if k=="pub" else f"{lo:g}"); end=lo
        w="bold" if k=="ours" else "normal"
        ax.text(end+xmax*0.012,y+0.08,val,va="center",color=FG if k!="plain" else DIM,fontsize=15,fontweight=w)
        ax.text(end+xmax*0.012,y-0.27,note,va="center",color=DIM,fontsize=9.5)
        ax.text(-xmax*0.02,y,lab,va="center",ha="right",color=FG if k in("ours","pub") else DIM,fontsize=12,fontweight=w)
    ax.set_xlim(0,xmax); ax.set_ylim(-0.6,n-0.4)
    ax.set_yticks([]); ax.tick_params(colors=DIM,labelsize=10)
    for s in ax.spines.values(): s.set_visible(False)
    ax.xaxis.grid(True,color="#1c242d",lw=1); ax.set_axisbelow(True)
    ax.set_title(title,color=FG,fontsize=17,fontweight="bold",loc="left",pad=12)
    ax.set_xlabel("tokens / second",color=DIM,fontsize=11)
panel(axes[0],single,"1 request",420)
panel(axes[1],multi,"4 requests, aggregate",900)

fig.text(0.02,0.935,"Lossless Qwen3.8-27B on one RTX 5090 — vs. published numbers",color=FG,fontsize=26,fontweight="bold")
fig.text(0.02,0.885,"Ours: W4A16 + DFlash2 K=7 · greedy = plain greedy of the same 4-bit checkpoint · batch-invariant (a 2nd request can't flip a token) · window 57,344 · 16 prompts, prefill included, replies not cut",
         color=DIM,fontsize=11.5)
leg=[Patch(color=OURS,label="Ours, locked 4 Nov"),Patch(color=OLD,label="Ours, older runs"),
     Patch(color=PLAIN,label="Ours, plain (no drafter)"),Patch(facecolor=PUB,hatch="///",edgecolor=BG,label="Published, different meter")]
fig.legend(handles=leg,loc="lower left",bbox_to_anchor=(0.02,0.075),ncol=4,frameon=False,fontsize=11.5,labelcolor=FG)
fig.text(0.02,0.03,"Not apples-to-apples. Published LukasParke fork: IQ4_XS, ~131k context, rejection sampling, short numbers often decode-only. Ours include prefill and run at a 57,344 window.\n"
         "Plain ~115 is a published decode of this model family. Our invariant plain kernels run 38 — the drafter is what buys the speed back. Sources: github.com/NeuralLoot/solomon-lossless-5090",
         color=DIM,fontsize=10,linespacing=1.5)
fig.savefig("/workspace/solomon-lossless-5090/assets/solomon-compare.png",facecolor=BG)
print("ok")
