# from ipdb import set_trace as br
# import sys, IPython; sys.excepthook = IPython.core.ultratb.ColorTB(call_pdb=True)
import glob, os, argparse
import numpy as np, pandas as pd, matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Random Portfolio Optimizer')
parser.add_argument('-s', '--start', type=str, default='20100101', help='start date of all time series')
parser.add_argument('-e', '--end', type=str, default=None, help='end date of all time series')
parser.add_argument('-N', '--trials', type=int, default=100000, help='max holdings for portfolio')
parser.add_argument('--minH', type=int, default=5, help='min holdings for portfolio')
parser.add_argument('--riskfree', type=float, default=0.025, help='risk-free interest rate')
parser.add_argument('--bestsharpe', type=int, default=25, help='if enabled, only use the best N stocks with the highest Sharpe ratio')
parser.add_argument('--plotlimit', type=int, default=250, help='max number of portfolio to plot')
parser.add_argument('--seed', type=int, default=1, help='random seed, same seed will yield same random result each time. change it if you want another set of random number')
sargs = parser.parse_args()
np.random.seed(sargs.seed)

class App:    
    def __init__(self, start=None, end=None, riskfree=0.025):     
        self.start, self.end = start, end
        self.riskfree = riskfree
        
    def prepare_data(self, bestsharpe=None):
        def get_close(path):
            ticker = os.path.basename(path).replace('.csv','')
            df = pd.read_csv(path, parse_dates=True, index_col=0)
            df = df.loc[self.start:self.end]
            series = df['Adj Close'].rename(ticker)
            return series
        prices = [get_close(path) for path in glob.glob('resources\\*.csv')]
        changes = [np.log(price).diff() for price in prices]
        max_valid_length = max([len(s.dropna()) for s in changes])
        valid_statistics = [change for change in changes if len(change.dropna())>0.95*max_valid_length]
        stats = pd.concat(valid_statistics, axis=1).dropna()        
        expRs = stats.mean()*250
        stds = stats.std()*np.sqrt(250)
        sharpes = (expRs-self.riskfree)/stds
        stock_metrics = pd.DataFrame({'expR':expRs, 'std':stds, 'sharpe':sharpes}
                                        ).sort_values(by='sharpe', ascending=False)
        self.stock_metrics = stock_metrics.iloc[:bestsharpe]
        self.stats = stats[self.stock_metrics.index]
        self.covM = self.stats.cov()*250
        self.tickers = self.stock_metrics.index

    def optimize(self, N=100, minH=5):
        def make_portfolio():
            size = np.random.randint(minH, len(self.tickers))
            holdings = np.random.random((size, ))
            pads = np.zeros(self.stats.shape[1]-size)
            weights = np.concatenate((holdings, pads))
            np.random.shuffle(weights)
            weights = weights/weights.sum()   
            return weights
        def porf_expR(weights):
            return np.dot(self.stock_metrics['expR'], weights)        
        def porf_std(weights):
            return np.sqrt(np.dot(np.dot(weights, self.covM), weights.T))
        def porf_metrics(weights):
            expR = porf_expR(weights)
            std = porf_std(weights)
            sharpe = (expR-self.riskfree)/std
            return {
                'porf': weights,
                'expR': expR,
                'std': std,
                'sharpe': sharpe
            }
        
        results = []
        required_sharpe = self.stock_metrics['sharpe'].max()
        for i in range(N):
            if i%100==0:
                print(f'{i: >6} / {N} | total {len(results): >5} results are found', end='\t\t\r')
            weights = make_portfolio()
            result = porf_metrics(weights)
            if result['sharpe']>required_sharpe:
                results.append(result)
        for result in results:
            holdings = pd.Series(result['porf'], index=self.tickers).sort_values(ascending=False)
            result['porf'] = holdings[holdings>0]
        
        self.results = pd.DataFrame(results).sort_values(by='sharpe', ascending=False)
        self.best = self.results.iloc[0]

    def plot(self, plotlimit=250):
        bporf, bexpR, bstd, bsharpe = self.best['porf'], self.best['expR'], self.best['std'], self.best['sharpe']

        fig, ax = plt.subplots(1,1, figsize=(15,10,) )   
        ax.axhline(0, c='k'); ax.axvline(0, c='k')
        ax.set_xlim(-0.05, +0.80); ax.set_ylim(-0.20, +0.80)     
        title = (f'Random Portfolio Generator\n'
                f'{self.stats.index[0].date()} - {self.stats.index[-1].date()}')
        ax.set_title(title)
        for ticker, stock in self.stock_metrics.iterrows():
            ax.scatter(stock['std'], stock['expR'], c='grey')
            ax.annotate(ticker, (stock['std'], stock['expR']), c='black')   
            if ticker in bporf.index:
                ax.scatter(stock['std'], stock['expR'], c='red', marker='^', s=100)
        for _, porf in self.results.iloc[:plotlimit].iterrows():
            ax.scatter(porf['std'], porf['expR'], c='blue', s=10)
        ax.scatter(bstd, bexpR, c='red', marker='X', s=200)
        info = (f"expR: {bexpR:.1%}, std: {bstd:.1%}\n"
            f"sharpe: {bsharpe:.1%}")
        holdings = '\n'.join([f'{t}: {pct:6.2%}' for t,pct in bporf.items() if pct>0])
        text = f'Best Random Portfolio:\n{info}\n\nHoldings:\n{holdings}'
        ax.annotate(text, (bstd, bexpR), c='blue',
            horizontalalignment='right', verticalalignment='bottom')
        plt.show()
        #br()

def main():
    app = App(
        start=sargs.start,
        end=sargs.end,
        riskfree=sargs.riskfree)
    app.prepare_data(
        bestsharpe=sargs.bestsharpe)
    app.optimize(
        N=sargs.trials,
        minH=sargs.minH)
    app.plot(
        plotlimit=sargs.plotlimit)

if __name__ == '__main__':
    main()