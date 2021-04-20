import xlsxwriter

workbook = xlsxwriter.Workbook('Results.xlsx')
worksheet = workbook.add_worksheet()

chart = workbook.add_chart({'type': 'bar'})

chart.add_series({
    'categories': '=Sheet1!$A$2:$A$4',
    'values':     '=Sheet1!$B$2:$B$4',
    'points': [
        {'fill': {'color': 'red'}},
        {'fill': {'color': 'green'}},
        {'fill': {'color': 'blue'}},
    ],
})

worksheet.insert_chart('B5', chart)
workbook.close()