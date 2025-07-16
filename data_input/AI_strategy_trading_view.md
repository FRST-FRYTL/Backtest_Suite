```pinescript
// This work is licensed under a Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/
// © LuxAlgo
// Strategy version by: AI Assistant

//@version=5
strategy("SuperTrend AI Strategy [LuxAlgo]", "LuxAlgo - SuperTrend AI Strategy", 
         overlay = true, 
         initial_capital = 10000,
         default_qty_type = strategy.percent_of_equity,
         default_qty_value = 100,
         commission_type = strategy.commission.percent,
         commission_value = 0.1,
         max_labels_count = 500)

//------------------------------------------------------------------------------
//Settings
//-----------------------------------------------------------------------------{
length = input(10, 'ATR Length')

minMult = input.int(1, 'Factor Range', minval = 0, inline = 'factor')
maxMult = input.int(5, '', minval = 0, inline = 'factor')
step    = input.float(.5, 'Step', minval = 0, step = 0.1)

//Trigger error
if minMult > maxMult
    runtime.error('Minimum factor is greater than maximum factor in the range')

perfAlpha = input.float(10, 'Performance Memory', minval = 2)
fromCluster = input.string('Best', 'From Cluster', options = ['Best', 'Average', 'Worst'])

//Strategy Settings
useSignalStrength = input.bool(true, 'Use Signal Strength Filter', group = 'Strategy Settings')
minSignalStrength = input.int(4, 'Minimum Signal Strength (0-10)', minval = 0, maxval = 10, group = 'Strategy Settings')
useTimeFilter = input.bool(false, 'Use Time Filter', group = 'Strategy Settings')
startHour = input.int(9, 'Start Hour', minval = 0, maxval = 23, group = 'Strategy Settings')
endHour = input.int(16, 'End Hour', minval = 0, maxval = 23, group = 'Strategy Settings')

//Risk Management
useStopLoss = input.bool(true, 'Use Stop Loss', group = 'Risk Management')
stopLossType = input.string('ATR', 'Stop Loss Type', options = ['ATR', 'Percentage'], group = 'Risk Management')
stopLossATR = input.float(2.0, 'Stop Loss ATR Multiplier', minval = 0.1, step = 0.1, group = 'Risk Management')
stopLossPerc = input.float(2.0, 'Stop Loss %', minval = 0.1, step = 0.1, group = 'Risk Management')

useTakeProfit = input.bool(true, 'Use Take Profit', group = 'Risk Management')
takeProfitType = input.string('Risk/Reward', 'Take Profit Type', options = ['Risk/Reward', 'ATR', 'Percentage'], group = 'Risk Management')
riskRewardRatio = input.float(2.0, 'Risk/Reward Ratio', minval = 0.1, step = 0.1, group = 'Risk Management')
takeProfitATR = input.float(3.0, 'Take Profit ATR Multiplier', minval = 0.1, step = 0.1, group = 'Risk Management')
takeProfitPerc = input.float(4.0, 'Take Profit %', minval = 0.1, step = 0.1, group = 'Risk Management')

//Optimization
maxIter = input.int(1000, 'Maximum Iteration Steps', minval = 0, group = 'Optimization')
maxData = input.int(10000, 'Historical Bars Calculation', minval = 0, group = 'Optimization')

//Style
bearCss = input(color.red, 'Trailing Stop', inline = 'ts', group = 'Style')
bullCss = input(color.teal, '', inline = 'ts', group = 'Style')

amaBearCss = input(color.new(color.red, 50), 'AMA', inline = 'ama', group = 'Style')
amaBullCss = input(color.new(color.teal, 50), '', inline = 'ama', group = 'Style')

showGradient = input(true, 'Candle Coloring', group = 'Style')
showSignals = input(true, 'Show Signals', group = 'Style')

//Dashboard
showDash  = input(true, 'Show Dashboard', group = 'Dashboard')
dashLoc  = input.string('Top Right', 'Location', options = ['Top Right', 'Bottom Right', 'Bottom Left'], group = 'Dashboard')
textSize = input.string('Small', 'Size'        , options = ['Tiny', 'Small', 'Normal'], group = 'Dashboard')

//-----------------------------------------------------------------------------}
//UDT's
//-----------------------------------------------------------------------------{
type supertrend
    float upper = hl2
    float lower = hl2
    float output
    float perf = 0
    float factor
    int trend = 0

type vector
    array<float> out

//-----------------------------------------------------------------------------}
//Supertrend
//-----------------------------------------------------------------------------{
var holder = array.new<supertrend>(0)
var factors = array.new<float>(0)

//Populate supertrend type array
if barstate.isfirst
    for i = 0 to int((maxMult - minMult) / step)
        factors.push(minMult + i * step)
        holder.push(supertrend.new())

atr = ta.atr(length)

//Compute Supertrend for multiple factors
k = 0
for factor in factors
    get_spt = holder.get(k)

    up = hl2 + atr * factor
    dn = hl2 - atr * factor
    
    get_spt.trend := close > get_spt.upper ? 1 : close < get_spt.lower ? 0 : get_spt.trend
    get_spt.upper := close[1] < get_spt.upper ? math.min(up, get_spt.upper) : up
    get_spt.lower := close[1] > get_spt.lower ? math.max(dn, get_spt.lower) : dn
    
    diff = nz(math.sign(close[1] - get_spt.output))
    get_spt.perf += 2/(perfAlpha+1) * (nz(close - close[1]) * diff - get_spt.perf)
    get_spt.output := get_spt.trend == 1 ? get_spt.lower : get_spt.upper
    get_spt.factor := factor
    k += 1

//-----------------------------------------------------------------------------}
//K-means clustering
//-----------------------------------------------------------------------------{
factor_array = array.new<float>(0)
data = array.new<float>(0)

//Populate data arrays
if last_bar_index - bar_index <= maxData
    for element in holder
        data.push(element.perf)
        factor_array.push(element.factor)

//Intitalize centroids using quartiles
centroids = array.new<float>(0)
centroids.push(data.percentile_linear_interpolation(25))
centroids.push(data.percentile_linear_interpolation(50))
centroids.push(data.percentile_linear_interpolation(75))

//Intialize clusters
var array<vector> factors_clusters = na
var array<vector> perfclusters = na

if last_bar_index - bar_index <= maxData
    for _ = 0 to maxIter
        factors_clusters := array.from(vector.new(array.new<float>(0)), vector.new(array.new<float>(0)), vector.new(array.new<float>(0)))
        perfclusters := array.from(vector.new(array.new<float>(0)), vector.new(array.new<float>(0)), vector.new(array.new<float>(0)))
        
        //Assign value to cluster
        i = 0
        for value in data
            dist = array.new<float>(0)
            for centroid in centroids
                dist.push(math.abs(value - centroid))

            idx = dist.indexof(dist.min())
            perfclusters.get(idx).out.push(value)
            factors_clusters.get(idx).out.push(factor_array.get(i))
            i += 1

        //Update centroids
        new_centroids = array.new<float>(0)
        for cluster_ in perfclusters
            new_centroids.push(cluster_.out.avg())

        //Test if centroid changed
        if new_centroids.get(0) == centroids.get(0) and new_centroids.get(1) == centroids.get(1) and new_centroids.get(2) == centroids.get(2)
            break

        centroids := new_centroids

//-----------------------------------------------------------------------------}
//Signals and trailing stop
//-----------------------------------------------------------------------------{
//Get associated supertrend
var float target_factor = na
var float perf_idx = na
var float perf_ama = na

var from = switch fromCluster
    'Best' => 2
    'Average' => 1
    'Worst' => 0

//Performance index denominator
den = ta.ema(math.abs(close - close[1]), int(perfAlpha))

if not na(perfclusters)
    //Get average factors within target cluster 
    target_factor := nz(factors_clusters.get(from).out.avg(), target_factor)
    
    //Get performance index of target cluster 
    perf_idx := math.max(nz(perfclusters.get(from).out.avg()), 0) / den

//Get new supertrend
var upper = hl2
var lower = hl2
var os = 0

up = hl2 + atr * target_factor
dn = hl2 - atr * target_factor
upper := close[1] < upper ? math.min(up, upper) : up
lower := close[1] > lower ? math.max(dn, lower) : dn
os := close > upper ? 1 : close < lower ? 0 : os
ts = os ? lower : upper

//Get trailing stop adaptive MA
if na(ts[1]) and not na(ts)
    perf_ama := ts
else
    perf_ama += perf_idx * (ts - perf_ama)

//-----------------------------------------------------------------------------}
//Strategy Logic
//-----------------------------------------------------------------------------{
// Calculate signal strength (0-10 scale)
signalStrength = int(perf_idx * 10)

// Time filter
timeInRange = not useTimeFilter or (hour >= startHour and hour <= endHour)

// Entry conditions
longCondition = os > os[1] and timeInRange
shortCondition = os < os[1] and timeInRange

// Apply signal strength filter if enabled
if useSignalStrength
    longCondition := longCondition and signalStrength >= minSignalStrength
    shortCondition := shortCondition and signalStrength >= minSignalStrength

// Entry logic
if longCondition and strategy.position_size <= 0
    strategy.entry("Long", strategy.long)
    
if shortCondition and strategy.position_size >= 0
    strategy.entry("Short", strategy.short)

// Exit logic - Trailing stop
if strategy.position_size > 0 and os < os[1]
    strategy.close("Long", comment="Trailing Stop")
    
if strategy.position_size < 0 and os > os[1]
    strategy.close("Short", comment="Trailing Stop")

// Risk Management - Stop Loss and Take Profit
if strategy.position_size != 0
    entryPrice = strategy.position_avg_price
    
    // Calculate stop loss
    stopLossLong = 0.0
    stopLossShort = 0.0
    
    if useStopLoss
        if stopLossType == "ATR"
            stopLossLong := entryPrice - (atr * stopLossATR)
            stopLossShort := entryPrice + (atr * stopLossATR)
        else // Percentage
            stopLossLong := entryPrice * (1 - stopLossPerc / 100)
            stopLossShort := entryPrice * (1 + stopLossPerc / 100)
    
    // Calculate take profit
    takeProfitLong = 0.0
    takeProfitShort = 0.0
    
    if useTakeProfit
        if takeProfitType == "Risk/Reward"
            riskLong = entryPrice - stopLossLong
            riskShort = stopLossShort - entryPrice
            takeProfitLong := entryPrice + (riskLong * riskRewardRatio)
            takeProfitShort := entryPrice - (riskShort * riskRewardRatio)
        else if takeProfitType == "ATR"
            takeProfitLong := entryPrice + (atr * takeProfitATR)
            takeProfitShort := entryPrice - (atr * takeProfitATR)
        else // Percentage
            takeProfitLong := entryPrice * (1 + takeProfitPerc / 100)
            takeProfitShort := entryPrice * (1 - takeProfitPerc / 100)
    
    // Apply exits
    if strategy.position_size > 0
        if useStopLoss
            strategy.exit("Long SL/TP", "Long", stop=stopLossLong, limit=useTakeProfit ? takeProfitLong : na)
    else if strategy.position_size < 0
        if useStopLoss
            strategy.exit("Short SL/TP", "Short", stop=stopLossShort, limit=useTakeProfit ? takeProfitShort : na)

//-----------------------------------------------------------------------------}
//Dashboard
//-----------------------------------------------------------------------------{
var table_position = dashLoc == 'Bottom Left' ? position.bottom_left 
  : dashLoc == 'Top Right' ? position.top_right 
  : position.bottom_right

var table_size = textSize == 'Tiny' ? size.tiny 
  : textSize == 'Small' ? size.small 
  : size.normal

var tb = table.new(table_position, 4, 5
  , bgcolor = #1e222d
  , border_color = #373a46
  , border_width = 1
  , frame_color = #373a46
  , frame_width = 1)

if showDash
    if barstate.isfirst
        tb.cell(0, 0, 'Cluster', text_color = color.white, text_size = table_size)
        tb.cell(0, 1, 'Best', text_color = color.white, text_size = table_size)
        tb.cell(0, 2, 'Average', text_color = color.white, text_size = table_size)
        tb.cell(0, 3, 'Worst', text_color = color.white, text_size = table_size)
        tb.cell(0, 4, 'Strategy', text_color = color.white, text_size = table_size)
    
        tb.cell(1, 0, 'Size', text_color = color.white, text_size = table_size)
        tb.cell(2, 0, 'Centroid', text_color = color.white, text_size = table_size)
        tb.cell(3, 0, 'Factors', text_color = color.white, text_size = table_size)
    
    if barstate.islast
        topN = perfclusters.get(2).out.size()
        midN = perfclusters.get(1).out.size()
        btmN = perfclusters.get(0).out.size()

        //Size
        tb.cell(1, 1, str.tostring(topN), text_color = color.white, text_size = table_size)
        tb.cell(1, 2, str.tostring(midN), text_color = color.white, text_size = table_size)
        tb.cell(1, 3, str.tostring(btmN), text_color = color.white, text_size = table_size)
        
        //Strategy Stats
        tb.cell(1, 4, 'Active: ' + fromCluster, text_color = color.white, text_size = table_size)
        tb.cell(2, 4, 'Signal: ' + str.tostring(signalStrength), text_color = color.white, text_size = table_size)
        tb.cell(3, 4, 'Position: ' + (strategy.position_size > 0 ? 'Long' : strategy.position_size < 0 ? 'Short' : 'Flat'), text_color = color.white, text_size = table_size)
        
        //Content
        tb.cell(3, 1, str.tostring(factors_clusters.get(2).out), text_color = color.white, text_size = table_size, text_halign = text.align_left)
        tb.cell(3, 2, str.tostring(factors_clusters.get(1).out), text_color = color.white, text_size = table_size, text_halign = text.align_left)
        tb.cell(3, 3, str.tostring(factors_clusters.get(0).out), text_color = color.white, text_size = table_size, text_halign = text.align_left)

        //Calculate dispersion around centroid
        i = 0
        for cluster_ in perfclusters
            disp = 0.
            if cluster_.out.size() > 1
                for value in cluster_.out
                    disp += math.abs(value - centroids.get(i))
            
            disp /= switch i
                0 => btmN
                1 => midN
                2 => topN

            i += 1
            tb.cell(2, 4 - i, str.tostring(disp, '#.####'), text_color = color.white, text_size = table_size)

//-----------------------------------------------------------------------------}
//Plots
//-----------------------------------------------------------------------------{
css = os ? bullCss : bearCss

plot(ts, 'Trailing Stop', os != os[1] ? na : css, linewidth=2)

plot(perf_ama, 'Trailing Stop AMA',
  ta.cross(close, perf_ama) ? na
  : close > perf_ama ? amaBullCss : amaBearCss, linewidth=1)

//Candle coloring
barcolor(showGradient ? color.from_gradient(perf_idx, 0, 1, color.new(css, 80), css) : na)

//Signals
n = bar_index

if showSignals
    if longCondition
        label.new(n, ts, str.tostring(signalStrength)
          , color = bullCss
          , style = label.style_label_up
          , textcolor = color.white
          , size = size.tiny)

    if shortCondition
        label.new(n, ts, str.tostring(signalStrength)
          , color = bearCss
          , style = label.style_label_down
          , textcolor = color.white
          , size = size.tiny)

// Plot stop loss and take profit levels when in position
if strategy.position_size != 0 and useStopLoss
    entryPrice = strategy.position_avg_price
    
    if strategy.position_size > 0
        stopLossLevel = stopLossType == "ATR" ? entryPrice - (atr * stopLossATR) : entryPrice * (1 - stopLossPerc / 100)
        line.new(bar_index[1], stopLossLevel, bar_index, stopLossLevel, color=color.red, style=line.style_dashed, width=1)
        
        if useTakeProfit
            takeProfitLevel = 0.0
            if takeProfitType == "Risk/Reward"
                risk = entryPrice - stopLossLevel
                takeProfitLevel := entryPrice + (risk * riskRewardRatio)
            else if takeProfitType == "ATR"
                takeProfitLevel := entryPrice + (atr * takeProfitATR)
            else
                takeProfitLevel := entryPrice * (1 + takeProfitPerc / 100)
            line.new(bar_index[1], takeProfitLevel, bar_index, takeProfitLevel, color=color.green, style=line.style_dashed, width=1)
    
    else if strategy.position_size < 0
        stopLossLevel = stopLossType == "ATR" ? entryPrice + (atr * stopLossATR) : entryPrice * (1 + stopLossPerc / 100)
        line.new(bar_index[1], stopLossLevel, bar_index, stopLossLevel, color=color.red, style=line.style_dashed, width=1)
        
        if useTakeProfit
            takeProfitLevel = 0.0
            if takeProfitType == "Risk/Reward"
                risk = stopLossLevel - entryPrice
                takeProfitLevel := entryPrice - (risk * riskRewardRatio)
            else if takeProfitType == "ATR"
                takeProfitLevel := entryPrice - (atr * takeProfitATR)
            else
                takeProfitLevel := entryPrice * (1 - takeProfitPerc / 100)
            line.new(bar_index[1], takeProfitLevel, bar_index, takeProfitLevel, color=color.green, style=line.style_dashed, width=1)

//-----------------------------------------------------------------------------}
```