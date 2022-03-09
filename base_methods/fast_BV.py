from cdt.causality.pairwise import BivariateFit 

class fastBV(BivariateFit):

    def predict_proba(self, dataset, **kwargs):
        a,b = dataset 
        l = min(a.shape[0], 500)
        return super().predict_proba((a[0:l], b[0:l]))
