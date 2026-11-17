"""Plot actual evaluations accompanying the uniform five-input certificate."""
from pathlib import Path
import os,csv,json
HERE=Path(__file__).resolve().parent
OUT=HERE/'same_measurement_interval'
os.environ['MPLCONFIGDIR']=str(HERE/'tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rows=list(csv.DictReader((OUT/'curve.csv').open()))
col=lambda key:np.array([float(r[key]) for r in rows])
cert=json.loads((OUT/'certificate.json').read_bytes())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'svg.fonttype':'none','figure.dpi':170})
fig,axes=plt.subplots(1,2,figsize=(9.2,3.5),layout='constrained')
q=100*col('Q')
axes[0].plot(q,col('R2'),label='Two local qubits',color='#2a6fbb',lw=1.8)
axes[0].plot(q,col('R3'),label='Three local qubits',color='#c65c21',lw=1.8)
axes[0].set(xlabel='QBER Q (%)',ylabel='Asymptotic secret bits per raw pair',
            title='(a) Same five-input measurement')
axes[0].legend(frameon=False)
axes[1].plot(q,100*(col('ratio')-1),color='#485b42',lw=1.8)
axes[1].axhline(100*(cert['rate_ratio_lower']-1),color='#485b42',ls='--',lw=1,
               label='Uniform certified lower bound')
axes[1].set(xlabel='QBER Q (%)',ylabel='Rate increase (%)',
            title='(b) Benefit from a lower raw-pair cost')
axes[1].legend(frameon=False,loc='lower right',fontsize=8)
for ax in axes: ax.grid(alpha=.15);ax.set_xlim(15,20)
fig.savefig(OUT/'same_measurement_interval.png',dpi=200)
fig.savefig(OUT/'same_measurement_interval.svg')
plt.close(fig)
print(OUT/'same_measurement_interval.png')
