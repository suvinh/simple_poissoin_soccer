#!/usr/bin/env python3

import datetime
import asyncio
import aiohttp
import requests as r
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, SoupStrainer
import urllib, json

# for run asyncio in jupyter / https://github.com/jupyter/notebook/issues/3397
import nest_asyncio
nest_asyncio.apply()

date_id = 'id13295'
fifa_url = 'https://www.fifa.com/api/ranking-overview'

def get_dates_html():
    # with HTMLSession() as session:
    #     r = session.get(f'{fifa_url}?dateId={date_id}/')
    #     r.html.render()
    # filter_tag = SoupStrainer("table", {"class": "fc-ranking-list-full_rankingTable__1u4hs"})
    # soup = BeautifulSoup(r.html.html, 'lxml', parse_only=filter_tag)
    # # print(soup)
    # dates = []
    # body = soup.find('tbody')
    # # print(body)
    # rows = body.find_all('tr')
    # for row in rows:
    #     cols = row.find_all('td')
    #     cols = [ele.text.strip() for ele in cols]
    #     dates.append([ele for ele in cols if ele])
    # print(dates)

    url = f'{fifa_url}?locale=en&dateId={date_id}'
    response = r.get(url)
    data_json = response.json()['rankings']
    # print(data_json['rankings'])
    dates = []
    for data in data_json:
        dates.append(data['rankingItem'])
        # print(data)
    print(dates)

    return dates


def create_dates_dataset(html_dates):
    date_ids = [li['data-value'] for li in html_dates]
    dates = [li.text.strip() for li in html_dates]
    dataset = pd.DataFrame(data={'date': dates, 'date_id': date_ids})
    
    # convert 'date' from str to datetime and sorting "old -> new"
    dataset['date'] = pd.to_datetime(dataset['date'], format='%d %B %Y')
    dataset.sort_values('date', ignore_index=True, inplace=True)
    assert dataset.date.min() == dataset.iloc[0].date, \
            "Incorrect dataset sorting"
    
    return dataset

dates_from_page = get_dates_html()
sys.exit()
dates_dataset = create_dates_dataset(dates_from_page)

assert len(dates_from_page) == dates_dataset.shape[0], \
        "Number of dates in html and dataset don't match"

async def get_rank_page(date_id, session):
    async with session.get(f'{fifa_url}?dateId={date_id}/') as response:
        page = await response.text()
        if response.status == 200:
            return {'page': page, 'id': date_id}
        else:
            print(f'Parse error, page: {response.url}')
            return False
        
        
def scrapy_rank_table(page, date):
    rows = BeautifulSoup(page, 
                          'html.parser', 
                          parse_only=SoupStrainer('tbody')).find_all('tr')
    table = []
    for row in rows:
        table.append({
            'id': int(row['data-team-id']), 
            'country_full': row.find('span', {'class': 'fi-t__nText'}).text, 
            'country_abrv': row.find('span', {'class': 'fi-t__nTri'}).text,
            'rank': int(row.find('td', {'class': 'fi-table__rank'}).text), 
            #'total_points': int(row.find('td', {'class': 'fi-table__points'}).text),
            'total_points': row.find('td', {'class': 'fi-table__points'}).text,
            #'previous_points': int(row.find('td', {'class': 'fi-table__prevpoints'}).text or 0),
            'previous_points': row.find('td', {'class': 'fi-table__prevpoints'}).text or 0,
            'rank_change': int(row.find('td', {'class': 'fi-table__rankingmovement'}).text.replace('-', '0')),
            'confederation': row.find('td', {'class': 'fi-table__confederation'}).text.strip('#'),
            'rank_date': date
        })
    return table
    

async def parse_ranks(pages_df):
    fifa_ranking = pd.DataFrame(columns=[
        'id', 'rank', 'country_full', 'country_abrv', 
        'total_points', 'previous_points', 'rank_change', 
        'confederation', 'rank_date'
    ])

    start_time = datetime.datetime.now()
    print("Start parsing.. ", datetime.datetime.now()-start_time)
    
    task_parse = []
    async with aiohttp.ClientSession() as session:
        for date_id in pages_df.date_id.to_list():
            task_parse += [asyncio.create_task(get_rank_page(date_id, session))]
    
        for task in asyncio.as_completed(task_parse):
            page = await task
            if not task:
                continue
            date_ranking = scrapy_rank_table(page['page'], 
                                             pages_df[pages_df.date_id == page['id']].date.iloc[0])
            fifa_ranking = fifa_ranking.append(date_ranking, ignore_index=True)

            if fifa_ranking.rank_date.nunique() % 50 == 0:
                print(f'Complite {fifa_ranking.rank_date.nunique()}/{pages_df.shape[0]} dates')
    
    fifa_ranking.sort_values('rank_date', ignore_index=True, inplace=True)
    print(f'Parsing complite. Time {datetime.datetime.now()-start_time}')
    return fifa_ranking


def data_correction(df):
    """ Handmade """
    # Lebanon has two abbreviations
    df.replace({'country_abrv': 'LIB'}, 'LBN', inplace=True)
    # Montenegro duplicates
    df.drop(df[df.id == 1903356].index, inplace=True)
    # North Macedonia has two full names
    df.replace({'country_full': 'FYR Macedonia'}, 'North Macedonia', inplace=True)
    # Cabo Verde has two full names
    df.replace({'country_full': 'Cape Verde Islands'}, 'Cabo Verde', inplace=True)
    # Saint Vincent and the Grenadines have two full names
    df.replace({'country_full': 'St. Vincent and the Grenadines'}, 'St. Vincent / Grenadines', inplace=True)
    # Swaziland has two full names
    df.replace({'country_full': 'Eswatini'}, 'Swaziland', inplace=True)
    # Curacao transform to Curaçao (with 'ç')
    df.replace({'country_full': 'Curacao'}, 'Curaçao', inplace=True)
    # São Tomé and Príncipe have three full names
    df.replace({'country_full': ['Sao Tome e Principe', 'São Tomé e Príncipe']}, 
               'São Tomé and Príncipe', inplace=True)
    return df


def check_data(ranks_df, dates_df):
    if ranks_df.rank_date.nunique() != dates_df.date.nunique():
        print("Warning! Numbers of rank dates don't match")
    if ranks_df.country_full.nunique() != ranks_df.country_abrv.nunique():
        print("Warning! Number of names and abbreviations does not match")
    if ranks_df.country_full.nunique() != ranks_df.id.nunique():
        print("Warning! Number of names and IDs does not match")
        

def save_as_csv(df):
    df.to_csv(
        f'fifa_ranking-{df.rank_date.max().date()}.csv',
        index=False, 
        encoding='utf-8'
    )
    print('Dataframe saved in currently folder')


fifa_ranking_df = asyncio.run(parse_ranks(dates_dataset))
fifa_ranking_df = data_correction(fifa_ranking_df)
check_data(fifa_ranking_df, dates_dataset)
save_as_csv(fifa_ranking_df)

fifa_ranking_df.tail()

teams = [['Italy', 'ITA', 'A'],
        ['Switzerland', 'SUI', 'A'],
        ['Turkey', 'TUR', 'A'],
        ['Wales', 'WAL', 'A'],
        ['Belgium', 'BEL', 'B'],
        ['Denmark', 'DEN', 'B'],
        ['Finland', 'FIN', 'B'],
        ['Russia', 'RUN', 'B'],
        ['Austria', 'AUT', 'C'],
        ['Netherlands', 'NED', 'C'],
        ['North Macedonia', 'MKD', 'C'],
        ['Ukraine', 'UKR', 'C'],
        ['Croatia', 'CRO', 'D'],
        ['Czech Republic', ' CZE', 'D'],
        ['England', 'ENG', 'D'],
        ['Scotland', 'SCO', 'D'],
        ['Poland', 'POL', 'E'],
        ['Slovakia', 'SVK', 'E'],
        ['Spain', 'ESP', 'E'],
        ['Sweden', 'SWE', 'E'],
        ['France', 'FRA', 'F'],
        ['Germany', 'GER', 'F'],
        ['Hungary', 'HUN', 'F'],
        ['Portugal', 'POR', 'F']]
teams_df = pd.DataFrame(teams, columns=['Country', 'Abrv', 'Group'])
teams_df.head(4)

match_df = pd.read_csv('results.csv')
# only matches from 1993
match_df = match_df[match_df.date > '1993-01-01']
# only matches between playing teams
match_df = match_df[match_df.home_team.isin(teams_df.Country.to_list())]
match_df = match_df[match_df.away_team.isin(teams_df.Country.to_list())]
match_df.reset_index(drop=True, inplace=True)
# drop city column
match_df.drop(labels='city', axis=1, inplace=True)
print('Number of matches after filtering: {}'.format(len(match_df)))


MATCHES = len(match_df)

home_rank = np.zeros(MATCHES, dtype=int)
away_rank = np.zeros(MATCHES, dtype=int)
home_total_points = np.zeros(MATCHES, dtype=float)
away_total_points = np.zeros(MATCHES, dtype=float)
for i in range(MATCHES):
    home_listing = fifa_ranking_df[((fifa_ranking_df.country_full == match_df.iloc[i].home_team) & 
                            (fifa_ranking_df.rank_date <= match_df.iloc[i].date))].sort_values(by='rank_date', ascending=False)
    
    try:
        home_rank[i] = int(home_listing.iloc[0]['rank'])
    except:
        home_rank[i] = 155
        
    away_listing = fifa_ranking_df[((fifa_ranking_df.country_full == match_df.iloc[i].away_team) & 
                            (fifa_ranking_df.rank_date <= match_df.iloc[i].date))].sort_values(by='rank_date', ascending=False)
        
    try:
        away_rank[i] = int(away_listing.iloc[0]['rank'])
    except:
        away_rank[i] = 155

match_df['home_rank'] = home_rank
match_df['away_rank'] = away_rank
match_df['friendly'] = (match_df.tournament == 'Friendly')
match_df.drop(labels=['tournament', 'date', 'country'], axis=1, inplace=True)
match_df.neutral = match_df.neutral.astype(int)
match_df.friendly = match_df.neutral.astype(int)
match_df.tail()

# print(match_df)

X = match_df[['home_team', 'away_team', 'neutral', 'home_rank', 'away_rank', 'friendly']]
y1 = match_df['home_score']
y2 = match_df['away_score']

onehot_columns = ['home_team', 'away_team']
onehot_df = X[onehot_columns]
onehot_df = pd.get_dummies(onehot_df, columns = onehot_columns)
match_onehot_drop = X.drop(onehot_columns, axis = 1)
match_onehot = pd.concat([match_onehot_drop, onehot_df], axis = 1)
match_onehot.head()


from xgboost import XGBRegressor

# home team score model
hmodel = XGBRegressor()
hmodel.fit(match_onehot.values, y1.values)
#away team score model
amodel = XGBRegressor()
amodel.fit(match_onehot.values, y2.values)

def predict(h_country, a_country, neutral=True):
    # create vector
    cols = ['neutral', 'home_rank', 'away_rank', 'friendly']
    for c in onehot_df.columns.to_list():
        cols.append(c)
    df = pd.DataFrame(np.zeros((1,len(cols)), dtype=int), columns=cols)
    if neutral:
        df.neutral.iloc[0] = 1
    else:
        df.neutral.iloc[0] = 0
    df.home_rank.iloc[0] = fifa_ranking_df[((fifa_ranking_df.rank_date == '2021-05-27') & (fifa_ranking_df.country_full == h_country))]['rank'].values[0]
    df.away_rank.iloc[0] = fifa_ranking_df[((fifa_ranking_df.rank_date == '2021-05-27') & (fifa_ranking_df.country_full == a_country))]['rank'].values[0]
    df['home_team_'+h_country].iloc[0] = 1
    df['away_team_'+a_country].iloc[0] = 1
    #df = df[hmodel.get_booster().feature_names]
    # predict
    hscore = int(hmodel.predict(df.iloc[0].to_numpy().reshape(1,52))[0])
    ascore = int(amodel.predict(df.iloc[0].to_numpy().reshape(1,52))[0])
    return hscore, ascore

while True:
    home_team = input("Home team: ")
    away_team = input("Away team: ")
    print(predict(home_team, away_team, True))
    cont = input("Do you wanna continue? ")
    if len(cont) == 0 or cont == "no":
        break