import os
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoServiceError
from azure.kusto.data.helpers import dataframe_from_result_table

KUSTO_CLUSTER = "https://azdeployer.kusto.windows.net/"
KUSTO_DATABASE = "AzDeployerKusto"
KUSTO_CLUSTER_CM = "https://azhealthstore.centralus.kusto.windows.net/"
KUSTO_DATABASE_CM = "AzureCM"
KUSTO_CLUSTER_ICM = "https://Icmcluster.kusto.windows.net/"
KUSTO_DATABASE_ICM = "IcmDataWarehouse"

class KustoData:
    def __init__(self, daysAgo, signalFilterFilePath, buildLabelQueryFilePath, skuTimeSeriesQueryFilePath):
        with open(signalFilterFilePath, 'r') as fin:
            self.signalFilterStr = " ".join(fin.readlines())

        with open(buildLabelQueryFilePath, 'r') as fin:
            self.buildLabelQueryStr = " ".join(fin.readlines()) % daysAgo

        with open(skuTimeSeriesQueryFilePath, 'r') as fin:
            self.skuTimeSeriesQueryStr = " ".join(fin.readlines())

        KCSB = KustoConnectionStringBuilder.with_interactive_login(
            KUSTO_CLUSTER)
        KUSTO_CLIENT = KustoClient(KCSB)

        KCSB_CM = KustoConnectionStringBuilder.with_interactive_login(
            KUSTO_CLUSTER_CM)
        KUSTO_CLIENT_CM = KustoClient(KCSB_CM)

        KCSB_ICM = KustoConnectionStringBuilder.with_interactive_login(
            KUSTO_CLUSTER_ICM)
        KUSTO_CLIENT_ICM = KustoClient(KCSB_ICM)

