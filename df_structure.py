{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd \n",
    "import json\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "json_structure = {'U11744134': {'AccountType': 'CORPORATION', 'Cushion': '1', 'LookAheadNextChange': '0', 'AccruedCash': '2.34', 'AvailableFunds': '32.67', 'BuyingPower': '32.67', 'EquityWithLoanValue': '32.67', 'ExcessLiquidity': '32.67', 'FullAvailableFunds': '32.67', 'FullExcessLiquidity': '32.67', 'FullInitMarginReq': '0.00', 'FullMaintMarginReq': '0.00', 'GrossPositionValue': '0.00', 'InitMarginReq': '0.00', 'LookAheadAvailableFunds': '32.67', 'LookAheadExcessLiquidity': '32.67', 'LookAheadInitMarginReq': '0.00', 'LookAheadMaintMarginReq': '0.00', 'MaintMarginReq': '0.00', 'NetLiquidation': '32.67', 'TotalCashValue': '30.33'}, 'U12930963': {'AccountType': 'CORPORATION', 'Cushion': '0.821433', 'LookAheadNextChange': '1720359000', 'AccruedCash': '0.00', 'AvailableFunds': '12291.43', 'BuyingPower': '81942.90', 'EquityWithLoanValue': '15344.99', 'ExcessLiquidity': '12607.47', 'FullAvailableFunds': '12291.43', 'FullExcessLiquidity': '12607.47', 'FullInitMarginReq': '3053.55', 'FullMaintMarginReq': '2740.67', 'GrossPositionValue': '11685.62', 'InitMarginReq': '3053.55', 'LookAheadAvailableFunds': '12291.43', 'LookAheadExcessLiquidity': '12607.47', 'LookAheadInitMarginReq': '3053.55', 'LookAheadMaintMarginReq': '2740.67', 'MaintMarginReq': '2740.67', 'NetLiquidation': '15348.14', 'TotalCashValue': '3662.52'}, 'U12940036': {'AccountType': 'CORPORATION', 'Cushion': '1', 'LookAheadNextChange': '1720359000', 'AccruedCash': '0.00', 'AvailableFunds': '12150.13', 'BuyingPower': '81000.86', 'EquityWithLoanValue': '13612.26', 'ExcessLiquidity': '12301.60', 'FullAvailableFunds': '12150.13', 'FullExcessLiquidity': '12301.60', 'FullInitMarginReq': '1462.13', 'FullMaintMarginReq': '1310.66', 'GrossPositionValue': '13581.28', 'InitMarginReq': '1462.13', 'LookAheadAvailableFunds': '12150.13', 'LookAheadExcessLiquidity': '12301.60', 'LookAheadInitMarginReq': '1462.13', 'LookAheadMaintMarginReq': '1310.66', 'MaintMarginReq': '1310.66', 'NetLiquidation': '13612.26', 'TotalCashValue': '2.66'}, 'U13231217': {'AccountType': 'CORPORATION', 'Cushion': '0.820707', 'LookAheadNextChange': '1720359000', 'AccruedCash': '0.00', 'AvailableFunds': '17862.22', 'BuyingPower': '119081.47', 'EquityWithLoanValue': '22250.68', 'ExcessLiquidity': '18261.77', 'FullAvailableFunds': '17862.22', 'FullExcessLiquidity': '18261.77', 'FullInitMarginReq': '4388.46', 'FullMaintMarginReq': '3989.51', 'GrossPositionValue': '6049.15', 'InitMarginReq': '4388.46', 'LookAheadAvailableFunds': '17862.22', 'LookAheadExcessLiquidity': '18261.77', 'LookAheadInitMarginReq': '4388.46', 'LookAheadMaintMarginReq': '3989.51', 'MaintMarginReq': '3989.51', 'NetLiquidation': '22251.28', 'TotalCashValue': '16202.13'}, 'U13254634': {'AccountType': 'CORPORATION', 'Cushion': '0.770422', 'LookAheadNextChange': '1720359000', 'AccruedCash': '0.00', 'AvailableFunds': '10637.29', 'BuyingPower': '70915.29', 'EquityWithLoanValue': '14231.28', 'ExcessLiquidity': '10964.32', 'FullAvailableFunds': '10637.29', 'FullExcessLiquidity': '10964.32', 'FullInitMarginReq': '3593.98', 'FullMaintMarginReq': '3267.26', 'GrossPositionValue': '10114.10', 'InitMarginReq': '3593.98', 'LookAheadAvailableFunds': '10637.29', 'LookAheadExcessLiquidity': '10964.32', 'LookAheadInitMarginReq': '3593.98', 'LookAheadMaintMarginReq': '3267.26', 'MaintMarginReq': '3267.26', 'NetLiquidation': '14231.58', 'TotalCashValue': '4117.48'}, 'U14452095': {'AccountType': 'CORPORATION', 'Cushion': '1', 'LookAheadNextChange': '1720366200', 'AccruedCash': '0.00', 'AvailableFunds': '7374.78', 'BuyingPower': '49165.22', 'EquityWithLoanValue': '10655.81', 'ExcessLiquidity': '8030.99', 'FullAvailableFunds': '7374.78', 'FullExcessLiquidity': '8030.99', 'FullInitMarginReq': '3281.02', 'FullMaintMarginReq': '2624.82', 'GrossPositionValue': '10499.28', 'InitMarginReq': '3281.02', 'LookAheadAvailableFunds': '7374.78', 'LookAheadExcessLiquidity': '8030.99', 'LookAheadInitMarginReq': '3281.02', 'LookAheadMaintMarginReq': '2624.82', 'MaintMarginReq': '2624.82', 'NetLiquidation': '10655.81', 'TotalCashValue': '156.53'}}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the corrected dictionary string\n",
    "\n",
    "# Convert the corrected dictionary to a DataFrame\n",
    "df_corrected = pd.DataFrame.from_dict(json_structure, orient='index')\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_corrected.to_csv(\"master_overview.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "{\n",
    "    \"U11744134\": \"CashACC\",\n",
    "    \"U12930963\": \"SPY3\",\n",
    "    \"U12940036\": \"SPY4\",\n",
    "    \"U13231217\": \"SPY5\",\n",
    "    \"U13254634\": \"GMO\",\n",
    "    \"U14452095\": \"HK\"\n",
    "}\n",
    "U12930963"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "sintro_venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.2"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
