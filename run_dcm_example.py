dataset = 'SUPPORT'
cv_folds = 5
prot_att = 'race'
fair_strategy = None
quantiles = [0.25, 0.5, 0.75]

from dcm.deep_cox_mixture import load_dataset
(x, t, e, a), folds, times = load_dataset(dataset, cv_folds, prot_att, fair_strategy, quantiles)

x = x.astype('float32')
t = t.astype('float32')
#%%

from dcm import dcm_tf as dcm

dataset = 'SUPPORT'
cv_folds = 5
prot_att = 'race'
fair_strategy = None
quantiles = [0.25, 0.5, 0.75]

from dcm.deep_cox_mixture import load_dataset
(x, t, e, a), folds, times = load_dataset(dataset, cv_folds, prot_att, fair_strategy, quantiles)

x = x.astype('float32')
t = t.astype('float32')
n = len(x)

tr_size = int(n * 0.70)
vl_size = int(n * 0.10)
te_size = int(n * 0.20)

x_train, x_test, x_val = x[:tr_size], x[-te_size:], x[tr_size:tr_size + vl_size]
t_train, t_test, t_val = t[:tr_size], t[-te_size:], t[tr_size:tr_size + vl_size]
e_train, e_test, e_val = e[:tr_size], e[-te_size:], e[tr_size:tr_size + vl_size]

k = 3
h = 100

model = dcm.DeepCoxMixture(k, h)

# We `train` the model for 50 epochs,
# with a learning rate of 1e-3,
# a batch size of 128 using the Adam optimizer.
model, losses = dcm.train(
    model,
    x_train,
    t_train,
    e_train,
    x_val,
    t_val,
    e_val,
    epochs=50,
    lr=1e-3,
    bs=128,
    use_posteriors=False,
    random_state=0,
    return_losses=True,
    patience=3
)

scores = []

for time in times:
    score = dcm.predict_scores(model, x_test, time)
    scores.append(score)

from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
import numpy as np

cis = []
brs = []

et_train = np.array(
    [(e_train[i], t_train[i]) for i in range(len(e_train))], dtype=[('e', bool), ('t', float)]
)
et_test = np.array(
    [(e_test[i], t_test[i]) for i in range(len(e_test))], dtype=[('e', bool), ('t', float)]
)
et_val = np.array(
    [(e_val[i], t_val[i]) for i in range(len(e_val))], dtype=[('e', bool), ('t', float)]
)

for i, _ in enumerate(times):
    cis.append(concordance_index_ipcw(et_train, et_test, 1 - scores[i], times[i])[0])

for i, _ in enumerate(times):
    brs.append(float(brier_score(et_train, et_test, scores[i], times[i])[1]))

roc_auc = []
for i, _ in enumerate(times):
    roc_auc.append(cumulative_dynamic_auc(et_train, et_test, 1 - scores[i], times[i])[0])
for quantile in enumerate(quantiles):
    print(f"For {quantile[1]} quantile,")
    print("TD Concordance Index:", cis[quantile[0]])
    print("Brier Score:", brs[quantile[0]])
    print("ROC AUC ", roc_auc[quantile[0]][0], "\n")
