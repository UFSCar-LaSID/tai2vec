import scripts as kw

def remove_single_interactions(df):
    while True:
        count_users = df[kw.COLUMN_USER_ID].value_counts(sort=False)
        count_items = df[kw.COLUMN_ITEM_ID].value_counts(sort=False)
        invalid_users = count_users[count_users==1].index
        invalid_items = count_items[count_items==1].index
        if len(invalid_users) == 0 and len(invalid_items) == 0:
            break
        df = df[(~df[kw.COLUMN_USER_ID].isin(invalid_users))&(~df[kw.COLUMN_ITEM_ID].isin(invalid_items))].copy()
    return df


def remove_cold_start(df_train, df_test):
    valid_users = df_test[kw.COLUMN_USER_ID].isin(df_train[kw.COLUMN_USER_ID])
    valid_items = df_test[kw.COLUMN_ITEM_ID].isin(df_train[kw.COLUMN_ITEM_ID])
    return df_test[(valid_users)&(valid_items)].copy()