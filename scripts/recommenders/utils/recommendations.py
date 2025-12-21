import pandas as pd
import numpy as np
import scripts as kw

def get_recommendations(df_train, df_test, item_item_sim, data_repr, k=kw.K, top_n=kw.TOP_N):
    """
    Generates recommendations for users based on a pre-computed sparse similarity list.
    This logic mirrors the pandas-based implementation.
    """
    target_users = df_test[kw.COLUMN_USER_ID].unique()
    
    # Merge user history with item similarities
    item_based_neighborhood = pd.merge(
        df_train[df_train[kw.COLUMN_USER_ID].isin(target_users)], 
        item_item_sim, 
        on=kw.COLUMN_ITEM_ID, 
        how='inner'
    )
    
    # Aggregate scores for candidate items
    final_sim = item_based_neighborhood.groupby([kw.COLUMN_USER_ID, 'neighbor'])['sim'].mean().reset_index()
    
    # Filter out items already seen by the user
    final_sim = final_sim.merge(
        df_train, 
        how='left', 
        left_on=[kw.COLUMN_USER_ID, 'neighbor'], 
        right_on=[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID]
    )
    final_sim = final_sim[final_sim[kw.COLUMN_ITEM_ID].isna()].drop(columns=[kw.COLUMN_ITEM_ID])
    
    # Get top N recommendations
    recommendations = final_sim.sort_values('sim', ascending=False).groupby(kw.COLUMN_USER_ID).head(top_n)
    
    # Rank the recommendations
    recommendations['rank'] = recommendations.groupby(kw.COLUMN_USER_ID).cumcount() + 1
    
    # Final formatting
    recommendations = recommendations.rename(columns={'neighbor': kw.COLUMN_ITEM_ID})
    recommendations = recommendations[[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, 'rank']].sort_values([kw.COLUMN_USER_ID, 'rank']).reset_index(drop=True)

    return recommendations
