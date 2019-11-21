import numpy as np
import tensorflow as tf

sess = tf.Session()

TYPE=np.float64

N = 1000000
data = np.random.normal(0, 1, N).astype(TYPE)
# Truncate data to make it harder
data = data[(data > -1) & (data < 5)]

# Define data as a variable so that it will be cached
X = tf.Variable(data, name='data')

mu = tf.Variable(TYPE(1), name='mu')
sigma = tf.Variable(TYPE(2), name='sigma')

def normal_log(X, mu, sigma):
    with tf.name_scope('normal_log') as scope:
      ret = tf.log(1 / (tf.constant(np.sqrt(2 * np.pi), dtype=TYPE) * sigma)) - tf.pow(X - mu, 2) / (tf.constant(2, dtype=TYPE) * tf.pow(sigma, 2))
    return ret

def trunc_log(X, left, right, logpdf, *args):
    '''
    Truncates `logpdf` so that it is limited to the
    region between `left` and `right`.
    '''
    with tf.name_scope('trunc_log') as scope:
        ret =  tf.exp(logpdf(X, *args))
        # Very primite integral
        N = TYPE(10000)
        x = tf.linspace(TYPE(left), TYPE(right), N, name='x_integration')
        integ = tf.reduce_sum(tf.exp(logpdf(x, *args)) * (right - left) / N)
        # Are we inside the region?
        inside = tf.logical_and(tf.greater_equal(X, left), tf.less_equal(X, right))
        # Return normalised logpdf if we're inside the region
        # Return zero outside of the region
        out = tf.select(inside, tf.log(ret / integ), tf.fill(tf.shape(ret), TYPE(-np.inf)))
    return out

nll = -tf.reduce_sum(trunc_log(X, -1, 5, normal_log, mu, sigma))
nll_hist = tf.histogram_summary('nll', nll)
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("logs", sess.graph_def)

init = tf.initialize_all_variables()
sess.run(init)

def func(mu_, sigma_):
    return sess.run(nll, feed_dict={ mu: mu_, sigma: sigma_ })

# Would be nice to use this
#
grads = tf.gradients(nll, [mu, sigma])
def grad(x):
    out = sess.run(grads, feed_dict={ mu: x[0], sigma: x[1] })
    return np.array(out)

from iminuit import Minuit

m = Minuit(func, mu_=10, sigma_=10, error_mu_=0.5, error_sigma_=0.5, limit_mu_=(-1, 100), limit_sigma_=(0, 100), errordef=1)
m.migrad()
m.minos()
mu_ = m.values['mu_']
sigma_ = m.values['sigma_']

import matplotlib.pyplot as plt
xs = np.linspace(-2, 6, 400)
plt.hist(data, normed=True, histtype='step', color='k', bins=200)
plt.plot(xs, np.exp(sess.run(trunc_log(X, -1, 5, normal_log, mu, sigma), feed_dict={mu:mu_, sigma:sigma_, X: xs})), 'b-')
plt.savefig('out.pdf')
print('Plot saved to `out.pdf`.')