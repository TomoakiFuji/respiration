# coding: UTF-8
'''
respiration.py

auther  : Tomoaki Fuji
history  : rev---+date----------+comment---------------------------------------------
           1.00  2017/12/13     初版作成
           1.01  2017/12/15     グラフへの描画オプションを追加
           1.02  2017/12/22     波形を0-1に正規化しないようにした
           1.03  2017/12/25     吸気開始点などの検出に用いたスムージング後の時系列データを
                                ファイル出力できるようにした
'''

import sys
import codecs
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os

# 分析対象について（適宜書き換える）------------------------------------------------------

# ファイル名（英数半角で）
input_file = 'respiration_data2.txt'

# グラフ描画（描画［手動ページめくり］：2, 描画［自動ページめくり］：1, 描画しない：0）
drawGraph = 2

# 各位置の検出処理（する：1, しない：0［吸気開始点などの検出はせずにグラフの描画のみ］）
detectPoint = 1

# しきい値A,B（破線）の描画（する：1, しない：0）
threshold_draw = 1
# Peak_areaの縦線描画（する：1, しない：0）drawGraph,detectPointが0のときは描画しない
Peak_area_draw = 0
# bottom_areaの縦線描画（する：1, しない：0）drawGraph,detectPointが0のときは描画しない
bottom_area_draw = 0
# ピーク位置の縦線描画（する：1, しない：0）drawGraph,detectPointが0のときは描画しない
peak_position_draw = 0
# 呼気開始時点の縦線描画（する：1, しない：0）drawGraph,detectPointが0のときは描画しない
respiration_Start_draw = 0

# 微分波形の描画（する：1, しない：0）drawGraphが0のときは描画しない
drawDiff = 1

# データ書き出し（する：1, しない：0）detectPointが0のときはこの値にかかわらずデータは書き出さない
writeData = 1

# graphフォルダへグラフ書き出し（する：1, しない：0） 注）drawGraph=0のときは無視
writeGraph = 0

# サンプリング周波数
samplingFreq = 1000

# 冒頭の呼吸データの読み飛ばし秒数（データ取得直後のデータの乱れ対策）
skipSecond = 0.5

# ヘッダ部読み飛ばし行数
ignoreRow = 7

# -----------------------------------------------------------------------------------

if detectPoint != 1:
    writeData = 0
    Peak_area_draw = 0
    bottom_area_draw = 0
    peak_position_draw = 0

filename, ext = os.path.splitext(input_file)
output_file = filename + '.csv'

samplingPeriod = 1.0 / samplingFreq
del samplingFreq

# ファイルからデータ読み込み
RPS = []
data_file = codecs.open(input_file, 'r', 'shift_jis')
i = 0
for line in data_file:
    i = i + 1
    line = line.rstrip()
    if i <= ignoreRow:
        continue
    RPS.append(float(line))
data_file.close()

RPS = np.array(RPS)

del ignoreRow, filename, data_file, i, line, input_file, ext

# 10個のデータを1個にする（サンプリング周波数1000Hzを100Hzに）
dataNum = 10
samplingPeriod = samplingPeriod * dataNum
dataSize = RPS.shape
dataSize = dataSize[0] // dataNum
RPS = np.resize(RPS, (dataSize, dataNum))
RPS = np.mean(RPS, axis=1)
del dataNum

# 時間データの作成（横軸）
time = np.arange(int(dataSize))
time = time * samplingPeriod

# データの冒頭指定秒数スキップ
RPS = np.delete(RPS, range(0, int(skipSecond / samplingPeriod)))
time = np.delete(time, range(0, int(skipSecond / samplingPeriod)))
del skipSecond

# 移動平均（論文に書いてある平滑化の前に実行；もと波形のノイズが多いため）
# n点で移動平均
n = 31  # 奇数で
w = np.ones(n)/n
RPS = np.convolve(RPS, w, mode='same')

# 前後(n-1)/2点を削除
n = n // 2
dataSize = RPS.shape
dataSize = dataSize[0]
RPS = np.delete(RPS, range(dataSize-n, dataSize))
RPS = np.delete(RPS, range(0, n))
time = np.delete(time, range(dataSize-n, dataSize))
time = np.delete(time, range(0, n))

# フィルター（重み付き移動平均）
n = 9
w = [1/13, 0, 3/13, 0, 5/13, 0, 3/13, 0, 1/13]
RPS = np.convolve(RPS, w, mode='same')

# 前後(n-1)/2点を削除
n = n // 2
dataSize = RPS.shape
dataSize = dataSize[0]
RPS = np.delete(RPS, range(dataSize-n, dataSize))
RPS = np.delete(RPS, range(0, n))
time = np.delete(time, range(dataSize-n, dataSize))
time = np.delete(time, range(0, n))
del n, w

# 0 - 1 に正規化
# min = np.min(RPS)
# max = np.max(RPS)
# RPS = (RPS-min)/(max-min)
# del min, max

# 差分
RPSshift = np.copy(RPS)
RPSshift = np.delete(RPSshift, range(0, 4))
RPSshift = np.append(RPSshift, np.zeros(4))

dRPS = RPSshift - RPS
del RPSshift

dataSize = dRPS.shape
dataSize = dataSize[0]
dRPS = np.delete(dRPS, range(dataSize-4, dataSize))

# 前後2点を削除
RPS = np.delete(RPS, range(dataSize-2, dataSize))
RPS = np.delete(RPS, range(0, 2))
time = np.delete(time, range(dataSize-2, dataSize))
time = np.delete(time, range(0, 2))

dataSize = RPS.shape
dataSize = dataSize[0]

# 30s区間が何個あるか。5個ならしきい値は 1-2, 2-3, 3-4-5 の3区間に分けて計算
# グラフ表示は 1-2, 2-3, 3-4, 4-5 の4区間
sectionNum = int(dataSize / 30.0 * samplingPeriod) + 1
# print(sectionNum)

if detectPoint != 0:
    ps = []
    Peak_area_End = []
    bottom_area_Start = []
    peak = []
    respiration_Start = []
    respiration_Mid = []
    serialNum = 1

for section in range(sectionNum):
    if section < sectionNum - 3:  # 最後の区間以外
        # 60s間ごとに計算（30sずつ重複）
        RPS_section = RPS[int(30.0 / samplingPeriod) * section:int(30.0 / samplingPeriod) * (section + 2)]
        dRPS_section = dRPS[int(30.0 / samplingPeriod) * section:int(30.0 / samplingPeriod) * (section + 2)]
        time_section = time[int(30.0 / samplingPeriod) * section:int(30.0 / samplingPeriod) * (section + 2)]
    else:
        RPS_section = RPS[int(30.0 / samplingPeriod) * section:np.size(RPS)]
        dRPS_section = dRPS[int(30.0 / samplingPeriod) * section:np.size(dRPS)]
        time_section = time[int(30.0 / samplingPeriod) * section:np.size(time)]

    # しきい値A,Bの計算
    # 標準偏差
    if section != sectionNum - 2:
        SD_dRPS = np.std(dRPS_section)
        thresholdA = SD_dRPS * 0.8
        thresholdB = SD_dRPS * 0.5
        del SD_dRPS

    if detectPoint != 0:
        ps.append([])
        Peak_area_End.append([])
        bottom_area_Start.append([])
        peak.append([])
        respiration_Start.append([])
        respiration_Mid.append([])
        N = 0

        # print(section)

        dataSize = RPS_section.shape
        dataSize = dataSize[0]

        if section == 0:
            i = int(1.5 / samplingPeriod)    # 1.5s 進めておく

            while dRPS_section[i] > 0:
                i = i + 1
            # 確実に負の位置となるようiを10インクリメント
            i = i + 10
        else:
            i = i - int(30.0 / samplingPeriod)

        while 1:
            flag = 0
            while dRPS_section[i] >= thresholdA:
                i = i + 1
                if i + 3.0 / samplingPeriod >= dataSize:
                    flag = 1
                    break
            if flag == 1:
                break

            flag = 0
            while dRPS_section[i] < thresholdA:
                i = i + 1   # ps(N)検出
                if i + 3.0 / samplingPeriod >= dataSize:
                    flag = 1
                    break
            if flag == 1:
                break

            ps[section].append(i + int(30.0 / samplingPeriod * section))
            Peak_area_End[section].append(i + int(30.0 / samplingPeriod * section + 3.0 / samplingPeriod))
            # ボトムエリアスタート位置は1.5s前
            bottom_area_Start[section].append(i + int(30.0 / samplingPeriod * section - 1.5 / samplingPeriod))
            # Peak_area(N)抽出
            dTmp = dRPS_section[ps[section][N]-int(30.0 / samplingPeriod * section):
                                Peak_area_End[section][N]-int(30.0 / samplingPeriod * section)]
            Tmp = RPS_section[ps[section][N]-int(30.0 / samplingPeriod * section):
                                Peak_area_End[section][N]-int(30.0 / samplingPeriod * section)]

            for j in range(0, int(3.0 / samplingPeriod)):
                if dTmp[j] < 0.0:
                    # 0.2s先まで全部負ならbreak
                    flag = 0
                    for k in range(1, int(0.2 / samplingPeriod)):
                        if j+k == int(3.0 / samplingPeriod - 1):
                            break
                        if dTmp[j+k] >= 0:
                            flag = 1
                            break
                    if flag == 0:
                        break
            # dTmp[j] が 微分値0の位置

            # 微分値0の位置がピーク値なら(10 / sampleFreq の誤差を認める)
            if abs(np.argmax(Tmp) - j) < 10:
                peak[section].append(i + int(30.0 / samplingPeriod * section) + np.argmax(Tmp))
            else:
                if j+10 >= 3.0 / samplingPeriod:
                    Tmp = Tmp[int(3.0 / samplingPeriod)]
                else:
                    Tmp = Tmp[j-10:j+10]
                peak[section].append(i + int(30.0 / samplingPeriod * section) + np.argmax(Tmp) + j - 10)

            # bottom_area(N)抽出
            dTmp = dRPS_section[bottom_area_Start[section][N]-int(30.0 / samplingPeriod * section):
                                ps[section][N]-int(30.0 / samplingPeriod * section)]
            Tmp = RPS_section[bottom_area_Start[section][N]-int(30.0 / samplingPeriod * section):
                                ps[section][N]-int(30.0 / samplingPeriod * section)]
            # ps(N)検出
            k = 0
            while dTmp[k] > thresholdB:
                k = k + 1
            while dTmp[k] < thresholdB:
                k = k + 1

            respiration_Start[section].append(bottom_area_Start[section][N]+k)

            # 直前のピークと吸気開始時点の値の中間値の時間を求める
            if N != 0:
                if peak[section][N - 1] >= respiration_Start[section][N]:
                    respiration_Mid[section].append(np.nan)
                else:
                    Tmp = RPS[peak[section][N - 1]:respiration_Start[section][N]]
                    midValue = (RPS[peak[section][N - 1]] + RPS[respiration_Start[section][N]]) / 2.0
                    Tmp = np.array(Tmp)
                    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
                    idx = np.abs(np.asarray(Tmp) - midValue).argmin()
                    respiration_Mid[section].append(peak[section][N - 1] + idx)
                    del idx
            elif (N == 0) & (section != 0):
                if peak[section - 1][len(peak[section - 1]) - 1] >= respiration_Start[section][N]:
                    respiration_Mid[section].append(np.nan)
                else:
                    Tmp = RPS[peak[section - 1][len(peak[section - 1]) - 1]:respiration_Start[section][N]]
                    midValue = (RPS[peak[section-1][len(peak[section-1])-1]] + RPS[respiration_Start[section][N]]) / 2.0
                    Tmp = np.array(Tmp)
                    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
                    idx = np.abs(np.asarray(Tmp) - midValue).argmin()
                    respiration_Mid[section].append(peak[section - 1][len(peak[section - 1]) - 1] + idx)
                    del idx

            i = peak[section][N] - int(30.0 / samplingPeriod * section) + 10  # 確実にピークを越えた場所（10インクリメントしておく）

            # 40sを超えていたら次へ
            if (time[peak[section][N]] > int(section * 30.0 + 40.0)) & (section != sectionNum - 1):
                break

            N = N + 1

    if drawGraph:
        # グラフ描画
        if drawDiff == 1:
            plt.figure(figsize=(13,4))
            ax1 = plt.subplot(2,1,1)
        else:
            plt.figure(figsize=(13,2))
            ax1 = plt.subplot(1,1,1)

        plt.ylabel('RPS')
        plt.plot(time_section, RPS_section, 'b')
        ax1.text(1.0, 1.0, str(section+1) + '/' + str(sectionNum), transform=ax1.transAxes,
                 verticalalignment='bottom', horizontalalignment='right')

        if detectPoint != 0:
            if section != 0:
                plt.plot(time[peak[section-1][len(peak[section-1])-1]], RPS[peak[section-1][len(peak[section-1])-1]],
                         'ro', markeredgecolor='#99000A')
                plt.plot(time[respiration_Start[section-1][len(respiration_Start[section-1])-1]],
                         RPS[respiration_Start[section-1][len(respiration_Start[section-1])-1]], 'o',
                         color='#00BF0C', markeredgecolor='#006607')
                plt.text(time[respiration_Start[section-1][len(respiration_Start[section-1])-1]]+0.2,
                              RPS[respiration_Start[section-1][len(respiration_Start[section-1])-1]],
                         serialNum-1, fontsize=8)

            for area in range(len(peak[section])-1):
                plt.plot(time[peak[section][area]], RPS[peak[section][area]],
                         'ro', markeredgecolor='#99000A')

            for area in range(len(respiration_Start[section])):
                plt.plot(time[respiration_Start[section][area]], RPS[respiration_Start[section][area]], 'o',
                         color='#00BF0C', markeredgecolor='#006607')
                plt.text(time[respiration_Start[section][area]]+0.2, RPS[respiration_Start[section][area]],
                         serialNum, fontsize=8)
                serialNum = serialNum + 1

            for area in range(len(respiration_Mid[section])):
                if np.isnan(respiration_Mid[section][area]) == False:
                    plt.plot(time[respiration_Mid[section][area]], RPS[respiration_Mid[section][area]],
                             'o', color='#FADF00', markeredgecolor='#7A6E00')

        if drawDiff == 1:
            ax2 = plt.subplot(2,1,2, sharex=ax1)
            plt.ylabel('dRPS')
            plt.xlabel('time [s]')
            plt.plot(time_section, dRPS_section, 'r')
            plt.axhline(y=0, lw=0.5, color='k')
            if threshold_draw == 1:
                plt.axhline(y=thresholdA, linestyle='dashed', lw=0.5, color='k')
                plt.axhline(y=thresholdB, linestyle='dotted', lw=0.5, color='k')

        # 軸の一覧取得
        axs = plt.gcf().get_axes()
        # 軸毎にループ
        for ax in axs:
            # 現在の軸を変更
            plt.axes(ax)
            if Peak_area_draw == 1:
                # 共通の縦線を描画
                if section != 0:
                    plt.axvline(x=time[ps[section-1][len(ps[section-1])-1]], lw=1.2, color='g')
                    plt.axvline(x=time[Peak_area_End[section-1][len(ps[section-1])-1]], lw=1.2, color='m')
                for area in range(len(ps[section]) - 1):
                    plt.axvline(x=time[ps[section][area]], lw=1.2, color='g')
                    plt.axvline(x=time[Peak_area_End[section][area]], lw=1.2, color='m')
            if peak_position_draw == 1:
                if section != 0:
                    plt.axvline(x=time[peak[section-1][len(peak[section-1])-1]], lw=1.2, color='k')
                for area in range(len(peak[section]) - 1):
                    plt.axvline(x=time[peak[section][area]], lw=1.2, color='k')
            if bottom_area_draw == 1:
                if section != 0:
                    plt.axvline(x=time[bottom_area_Start[section-1][len(bottom_area_Start[section-1])-1]],
                                lw=1.2, color='y')
                for area in range(len(bottom_area_Start[section]) - 1):
                    plt.axvline(x=time[bottom_area_Start[section][area]], lw=1.2, color='y')
            if respiration_Start_draw == 1:
                if section != 0:
                    plt.axvline(x=time[respiration_Start[section-1][len(respiration_Start[section-1])-1]],
                                lw=1.2, color='gray')
                for area in range(len(bottom_area_Start[section])):
                    plt.axvline(x=time[respiration_Start[section][area]], lw=1.2, color='gray')

        # ax1.set_ylim(-0.05, 1.05)
        ax1.set_ylim(min(RPS)-3, max(RPS)+3)
        ax1.set_xlim([section * 30, (section + 2) * 30])
        ax1.set_xticks([section * 30, section * 30 + 5, section * 30 + 10, section * 30 + 15, section * 30 + 20,
                        section * 30 + 25, section * 30 + 30, section * 30 + 35, section * 30 + 40, section * 30 + 45,
                        section * 30 + 50, section * 30 + 55, section * 30 + 60])

        if writeGraph:
            if os.path.isdir('graph') == False:
                os.mkdir('graph')
            filename_graph = 'graph/graph_' + '{0:03d}'.format(section+1)
            plt.savefig(filename_graph)

        if drawGraph == 1:
            plt.pause(0.01)
        else:
            plt.show()

if writeData:
    respiration_Start_tmp = []
    for section in respiration_Start:
        for n in section:
            respiration_Start_tmp.append(n)
    respiration_Start = respiration_Start_tmp
    del respiration_Start_tmp

    respiration_Mid_tmp = []
    for section in respiration_Mid:
        for n in section:
            respiration_Mid_tmp.append(n)
    respiration_Mid = respiration_Mid_tmp
    del respiration_Mid_tmp

    peak_tmp = []
    for section in peak:
        for n in section:
            peak_tmp.append(n)
    peak = peak_tmp
    del peak_tmp

    f = open(output_file, 'w')

    f.write('通番,'
            ' 吸気開始時点の時間[s], そのときの値,'
            ' 吸気終了時点の時間[s], そのときの値,'
            ' 50%呼出時点の時間[s], そのときの値,'
            ' 次の吸気開始時点の時間[s], そのときの値\n')

    for i in range(len(respiration_Start)-1):
        if np.isnan(respiration_Mid[i]):
            continue
        f.write("{0:4d}".format(i+1)
                + ', '
                + '%7.2f' % time[respiration_Start[i]]
                + ', '
                + '%7.2f' % RPS[respiration_Start[i]]
                + ', '
                + '%7.2f' % time[peak[i]]
                + ', '
                + '%7.2f' % RPS[peak[i]]
                + ', '
                + '%7.2f' % time[respiration_Mid[i]]
                + ', '
                + '%7.2f' % RPS[respiration_Mid[i]]
                + ', '
                + '%7.2f' % time[respiration_Start[i+1]]
                + ', '
                + '%7.2f' % RPS[respiration_Start[i+1]]
                + '\n')

    f.close()