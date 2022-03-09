from cdt.causality.pairwise import ANM 

class fastANM(ANM):

    def predict_proba(self, data, **kwargs):
        a,b = data 
        l = min(a.shape[0], 500)
        return super().predict_proba((a[0:l], b[0:l]))
