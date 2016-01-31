import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from pandas.tools.plotting import autocorrelation_plot
import math

df = pd.read_csv('LoanStats3a.csv', skiprows=1, low_memory = False)
df['issue_d_format'] = pd.to_datetime(df['issue_d']) 
dfts = df.set_index('issue_d_format') 
year_month_summary = dfts.groupby(lambda x : x.year * 100 + x.month).count()
loan_count_summary = year_month_summary['issue_d']

#returns acf coefficients in a list http://stackoverflow.com/questions/14297012/estimate-autocorrelation-using-python
def acf(series):
    n = len(series)
    data = np.asarray(series)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
        return round(acf_lag, 3)
    x = np.arange(n) # Avoiding lag 0 calculation
    acf_coeffs = map(r, x)
    return acf_coeffs

def main():
	sm.graphics.tsa.plot_acf(loan_count_summary)
	sm.graphics.tsa.plot_pacf(loan_count_summary)
	plt.show()
	
	#tests list of acf coefficients for values above or below 95% CI level http://www.ltrr.arizona.edu/~dmeko/notes_3.pdf
	def testing():
		upper = 2/(math.sqrt(loan_count_summary.count()))
		lower = -2/(math.sqrt(loan_count_summary.count()))
		for elem in acf(loan_count_summary):
    			if (lower >= elem or elem >= upper):
        			return True

	#prints relevant message
	def printing():
		if testing() == True:
			print('Autocorrelated structures found')
		else:
			print('No autocorrelated structures found')
            
	printing()

if __name__ == "__main__":
    main()


