from caserec.recommenders.item_recommendation.content_based import ContentBased
from caserec.recommenders.item_recommendation.item_attribute_knn import ItemAttributeKNN

train = 'C:/datasets/ml-100k/train.dat'
test = 'C:/datasets/ml-100k/test.dat'
rank_cb = 'C:/datasets/ml-100k/rank_cb.dat'
rank_attr = 'C:/datasets/ml-100k/rank_attr.dat'
similarity1 = 'C:/datasets/ml-100k/sim_coo_vsm_new_version.dat'
similarity2 = 'C:/datasets/ml-100k/sim_ldsd_iw_new_version.dat'
similarity3 = 'C:/datasets/ml-100k/sim_vsm_new_version.dat'

metrics = ('PREC', 'RECALL', 'NDCG', 'MAP')

print(similarity1)
# ItemAttributeKNN(train, test, similarity_file=similarity1, output_file=rank_attr, rank_length=50).\
#     compute(metrics=metrics, n_ranks=[10, 20])

ContentBased(train, test, similarity_file=similarity1, output_file=rank_cb, rank_length=50).\
   compute(metrics=metrics, n_ranks=[10, 20, 50])

#print(similarity2)
#ItemAttributeKNN(train, test, similarity_file=similarity2, output_file=rank_attr, rank_length=50).\
#    compute(metrics=metrics, n_ranks=[10, 20])

#ContentBased(train, test, similarity_file=similarity2, output_file=rank_cb, rank_length=50).\
#   compute(metrics=metrics, n_ranks=[10, 20, 50])

#print(similarity3)
#ContentBased(train, test, similarity_file=similarity3, output_file=rank_cb, rank_length=50).\
#   compute(metrics=metrics, n_ranks=[10, 20, 50])

