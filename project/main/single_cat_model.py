import pandas as pd
import numpy as np
from datetime import datetime
from catboost import CatBoostRegressor, sum_models
import re
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings


warnings.filterwarnings('ignore')

train_columns = ['sold_price', 'price', 'view_count', 'steam_item_id', 'steam_cs2_profile_rank', 'steam_level', 'steam_friend_count', 'steam_dota2_solo_mmr', 'steam_converted_balance', 'steam_pubg_inv_value', 'steam_cs2_inv_value', 'steam_dota2_inv_value', 'steam_tf2_inv_value', 'steam_rust_inv_value', 'steam_game_count', 'steam_steam_inv_value', 'steam_inv_value', 'steam_cs2_win_count', 'steam_dota2_game_count', 'steam_dota2_lose_count', 'steam_dota2_win_count', 'steam_hours_played_recently', 'steam_faceit_level', 'steam_points', 'steam_relevant_game_count', 'steam_gift_count', 'steam_dota2_behavior', 'steam_mfa', 'steam_market', 'steam_market_restrictions', 'steam_unturned_inv_value', 'steam_kf2_inv_value', 'steam_dst_inv_value', 'steam_cs2_premier_elo', 'steam_rust_kill_player', 'steam_rust_deaths', 'steam_total_gifts_rub', 'steam_total_refunds_rub', 'steam_total_ingame_rub', 'steam_total_games_rub', 'steam_total_purchased_rub', 'steam_dota2_last_match_date', 'published_date_year', 'published_date_month', 'published_date_day', 'published_date_hour', 'published_date_minute', 'published_date_second', 'published_date_weekday', 'update_stat_date_year', 'update_stat_date_month', 'update_stat_date_day', 'update_stat_date_hour', 'update_stat_date_minute', 'update_stat_date_second', 'update_stat_date_weekday', 'refreshed_date_year', 'refreshed_date_month', 'refreshed_date_day', 'refreshed_date_hour', 'refreshed_date_minute', 'refreshed_date_second', 'refreshed_date_weekday', 'steam_register_date_year', 'steam_register_date_month', 'steam_register_date_day', 'steam_register_date_hour', 'steam_register_date_minute', 'steam_register_date_second', 'steam_register_date_weekday', 'steam_last_activity_year', 'steam_last_activity_month', 'steam_last_activity_day', 'steam_last_activity_hour', 'steam_last_activity_minute', 'steam_last_activity_second', 'steam_last_activity_weekday', 'steam_cs2_last_activity_year', 'steam_cs2_last_activity_month', 'steam_cs2_last_activity_day', 'steam_cs2_last_activity_hour', 'steam_cs2_last_activity_minute', 'steam_cs2_last_activity_second', 'steam_cs2_last_activity_weekday', 'steam_cs2_ban_date_year', 'steam_cs2_ban_date_month', 'steam_cs2_ban_date_day', 'steam_cs2_ban_date_hour', 'steam_cs2_ban_date_minute', 'steam_cs2_ban_date_second', 'steam_cs2_ban_date_weekday', 'steam_last_transaction_date_year', 'steam_last_transaction_date_month', 'steam_last_transaction_date_day', 'steam_last_transaction_date_hour', 'steam_last_transaction_date_minute', 'steam_last_transaction_date_second', 'steam_last_transaction_date_weekday', 'steam_market_ban_end_date_year', 'steam_market_ban_end_date_month', 'steam_market_ban_end_date_day', 'steam_market_ban_end_date_hour', 'steam_market_ban_end_date_minute', 'steam_market_ban_end_date_second', 'steam_market_ban_end_date_weekday', 'steam_cs2_last_launched_year', 'steam_cs2_last_launched_month', 'steam_cs2_last_launched_day', 'steam_cs2_last_launched_hour', 'steam_cs2_last_launched_minute', 'steam_cs2_last_launched_second', 'steam_cs2_last_launched_weekday', 'inv_value_sum', 'game_count_sum', 'level_sum', 'total_playtime', 'average_playtime', 'std_playtime', 'num_games_played', 'num_games_zero_playtime', 'total_steam_games', 'price_per_view', 'item_origin_autoreg', 'item_origin_brute', 'item_origin_fishing', 'item_origin_personal', 'item_origin_resale', 'item_origin_stealer', 'extended_guarantee_-1', 'extended_guarantee_0', 'extended_guarantee_1', 'nsb_-1', 'nsb_0', 'nsb_1', 'email_type_', 'email_type_native', 'item_domain_', 'item_domain_acioumail.com', 'item_domain_acrometermail.com', 'item_domain_actiodmail.com', 'item_domain_adoenmail.com', 'item_domain_adventuredmail.com', 'item_domain_alounmail.com', 'item_domain_amentarianmail.com', 'item_domain_amwomail.com', 'item_domain_antimimail.com', 'item_domain_antiscmail.com', 'item_domain_aplotmail.com', 'item_domain_athemail.com', 'item_domain_ationismmail.com', 'item_domain_ationsmail.com', 'item_domain_atoptomail.com', 'item_domain_autisdmail.com', 'item_domain_autorambler.ru', 'item_domain_basisdmail.com', 'item_domain_belieomail.com', 'item_domain_berkensmail.com', 'item_domain_bfirstmail.com', 'item_domain_bk.ru', 'item_domain_boredemail.com', 'item_domain_boreemail.com', 'item_domain_bsenmail.com', 'item_domain_buffsmail.com', 'item_domain_bundrpmail.com', 'item_domain_canismmail.com', 'item_domain_caramail.com', 'item_domain_catachremail.com', 'item_domain_catimail.com', 'item_domain_cbbomail.com', 'item_domain_chalcomail.com', 'item_domain_chiroptmail.com', 'item_domain_cialidsmail.com', 'item_domain_cidenmail.com', 'item_domain_cirefusmail.com', 'item_domain_closedemail.com', 'item_domain_collamailder.com', 'item_domain_confinmail.com', 'item_domain_creadmail.com', 'item_domain_creasmail.com', 'item_domain_ctabilitymail.com', 'item_domain_dactyliamail.com', 'item_domain_dendrochromail.com', 'item_domain_denumail.com', 'item_domain_desolamail.com', 'item_domain_despmail.com', 'item_domain_dfirstmail.com', 'item_domain_diencmail.com', 'item_domain_docummail.com', 'item_domain_drologymail.com', 'item_domain_ecidmail.com', 'item_domain_elvemail.com', 'item_domain_email.com', 'item_domain_emocmail.com', 'item_domain_emunemail.com', 'item_domain_eoantmail.com', 'item_domain_epicamail.com', 'item_domain_eriesemail.com', 'item_domain_erizationmail.com', 'item_domain_erviomail.com', 'item_domain_ervmail.com', 'item_domain_eschaumail.com', 'item_domain_espismail.com', 'item_domain_ethodmail.com', 'item_domain_firepmail.com', 'item_domain_firstmailler.com', 'item_domain_firstmailler.net', 'item_domain_fixedmail.pw', 'item_domain_fmaild.com', 'item_domain_fontimail.com', 'item_domain_forecamail.com', 'item_domain_foresmail.com', 'item_domain_frescmail.com', 'item_domain_gazeta.pl', 'item_domain_globodmail.com', 'item_domain_gmx.at', 'item_domain_gmx.ch', 'item_domain_gmx.com', 'item_domain_gmx.de', 'item_domain_gmx.net', 'item_domain_handymail.pw', 'item_domain_happedmail.com', 'item_domain_healtmail.com', 'item_domain_hiccupemail.com', 'item_domain_hot.ee', 'item_domain_hotmail.com', 'item_domain_hotmail.fr', 'item_domain_hurcmail.com', 'item_domain_hydroxytrmail.com', 'item_domain_iddulmail.com', 'item_domain_iderdmail.com', 'item_domain_idiopmail.com', 'item_domain_igeonmail.com', 'item_domain_ignomail.com', 'item_domain_ikebremail.com', 'item_domain_inafmail.com', 'item_domain_inationalmail.com', 'item_domain_inbox.ru', 'item_domain_indecimail.com', 'item_domain_ineiuomail.com', 'item_domain_inemimail.com', 'item_domain_inisationmail.com', 'item_domain_inseimail.com', 'item_domain_insignimail.com', 'item_domain_interdmail.com', 'item_domain_internet.ru', 'item_domain_intrafomail.com', 'item_domain_invaspmail.com', 'item_domain_irectionalmail.pw', 'item_domain_itarilymail.com', 'item_domain_itepmail.com', 'item_domain_japadmail.com', 'item_domain_jlchacha.com', 'item_domain_jumimail.com', 'item_domain_jungledmail.com', 'item_domain_kaputmail.pw', 'item_domain_lactclmail.com', 'item_domain_legalmail.pw', 'item_domain_lenta.ru', 'item_domain_lihemail.com', 'item_domain_list.ru', 'item_domain_llusimail.com', 'item_domain_lustumail.com', 'item_domain_lyticallymail.com', 'item_domain_mail.com', 'item_domain_mail.ru', 'item_domain_margimail.com', 'item_domain_maturmail.com', 'item_domain_mbryonatemail.com', 'item_domain_meditekmail.com', 'item_domain_meningmail.com', 'item_domain_mergencmail.com', 'item_domain_meta.ua', 'item_domain_metachmail.com', 'item_domain_methylamail.com', 'item_domain_monodmail.com', 'item_domain_mosaimail.com', 'item_domain_motivamail.com', 'item_domain_muggmail.com', 'item_domain_myrambler.ru', 'item_domain_nickymail.com', 'item_domain_noissmail.com', 'item_domain_nologicalmail.com', 'item_domain_nonphilomail.com', 'item_domain_nsicalitymail.com', 'item_domain_nterhymail.com', 'item_domain_ntiarymail.com', 'item_domain_nvestmail.com', 'item_domain_o2.pl', 'item_domain_ocardiamail.com', 'item_domain_odulonmail.com', 'item_domain_oetrymail.com', 'item_domain_ogenesismail.com', 'item_domain_ographicallymail.com', 'item_domain_olographicmail.com', 'item_domain_olononmail.com', 'item_domain_ombemail.com', 'item_domain_ompensationmail.com', 'item_domain_onet.pl', 'item_domain_oneuralgiamail.com', 'item_domain_onsensmail.com', 'item_domain_op.pl', 'item_domain_operculummail.com', 'item_domain_opiniomail.com', 'item_domain_orange.fr', 'item_domain_otrimail.com', 'item_domain_oughlmail.com', 'item_domain_outlook.com', 'item_domain_outlook.es', 'item_domain_packamail.com', 'item_domain_palamail.com', 'item_domain_panionablemail.com', 'item_domain_pdaonmail.com', 'item_domain_photnsimail.com', 'item_domain_pimeralmail.pw', 'item_domain_piricismmail.com', 'item_domain_polidemail.com', 'item_domain_polisonmail.com', 'item_domain_polygmail.com', 'item_domain_precemail.com', 'item_domain_predamail.com', 'item_domain_protecemail.com', 'item_domain_provetmail.com', 'item_domain_pseudocmail.com', 'item_domain_quartmail.com', 'item_domain_quasdmail.com', 'item_domain_queammail.com', 'item_domain_racioumail.com', 'item_domain_rambler.ru', 'item_domain_rancemail.com', 'item_domain_rattymail.pw', 'item_domain_raymanmail.com', 'item_domain_rcardiogrammail.com', 'item_domain_recognimail.com', 'item_domain_redevtmail.com', 'item_domain_remilimail.com', 'item_domain_repormail.com', 'item_domain_retaipmail.com', 'item_domain_retirmail.com', 'item_domain_ro.ru', 'item_domain_roideamail.com', 'item_domain_romatomail.com', 'item_domain_rotesqumail.com', 'item_domain_rudgmail.com', 'item_domain_scalomail.com', 'item_domain_semicemail.com', 'item_domain_sertemail.com', 'item_domain_seumnmail.com', 'item_domain_seznam.cz', 'item_domain_sfirstmail.com', 'item_domain_sfr.fr', 'item_domain_sheelmail.com', 'item_domain_siogenicmail.com', 'item_domain_slippmail.com', 'item_domain_slummail.com', 'item_domain_solutiomail.com', 'item_domain_sordidmail.com', 'item_domain_spitalitmail.com', 'item_domain_ssociamail.com', 'item_domain_steninmail.com', 'item_domain_stewamail.com', 'item_domain_supersclmail.com', 'item_domain_survedmail.com', 'item_domain_swiftmail.pw', 'item_domain_t-online.de', 'item_domain_talenmail.com', 'item_domain_televsmail.com', 'item_domain_telkomsa.net', 'item_domain_temermail.com', 'item_domain_temporommail.com', 'item_domain_teractimail.com', 'item_domain_tesqumail.com', 'item_domain_tformemail.com', 'item_domain_tlen.pl', 'item_domain_tograpmail.com', 'item_domain_tsreademail.com', 'item_domain_ttlebmail.com', 'item_domain_ukr.net', 'item_domain_ulturmail.com', 'item_domain_unconsmail.com', 'item_domain_undldmail.com', 'item_domain_untenmail.com', 'item_domain_urringmail.com', 'item_domain_usemail.online', 'item_domain_vantagednessmail.com', 'item_domain_verticamail.com', 'item_domain_web.de', 'item_domain_wirpmail.com', 'item_domain_wp.pl', 'item_domain_xceemail.com', 'item_domain_xtenmail.com', 'resale_item_origin_', 'resale_item_origin_autoreg', 'resale_item_origin_brute', 'resale_item_origin_fishing', 'resale_item_origin_personal', 'resale_item_origin_resale', 'resale_item_origin_stealer', 'steam_country_A1', 'steam_country_Aland Islands', 'steam_country_Albania', 'steam_country_Algeria', 'steam_country_American Samoa', 'steam_country_Andorra', 'steam_country_Angola', 'steam_country_Anguilla', 'steam_country_Antigua and Barbuda', 'steam_country_Argentina', 'steam_country_Armenia', 'steam_country_Aruba', 'steam_country_Australia', 'steam_country_Austria', 'steam_country_Azerbaijan', 'steam_country_Bahamas', 'steam_country_Bahrain', 'steam_country_Bangladesh', 'steam_country_Barbados', 'steam_country_Belarus', 'steam_country_Belgium', 'steam_country_Belize', 'steam_country_Bhutan', 'steam_country_Bolivia', 'steam_country_Bosnia and Herzegovina', 'steam_country_Botswana', 'steam_country_Brazil', 'steam_country_Brunei Darussalam', 'steam_country_Bulgaria', 'steam_country_Cambodia', 'steam_country_Cameroon', 'steam_country_Canada', 'steam_country_Cape Verde', 'steam_country_Chile', 'steam_country_China', 'steam_country_Colombia', 'steam_country_Costa Rica', "steam_country_Cote d'Ivoire", 'steam_country_Croatia', 'steam_country_Cyprus', 'steam_country_Czech Republic', 'steam_country_Denmark', 'steam_country_Dominica', 'steam_country_Dominican Republic', 'steam_country_Ecuador', 'steam_country_Egypt', 'steam_country_El Salvador', 'steam_country_Estonia', 'steam_country_Faroe Islands', 'steam_country_Finland', 'steam_country_France', 'steam_country_French Guiana', 'steam_country_French Polynesia', 'steam_country_Gabon', 'steam_country_Georgia', 'steam_country_Germany', 'steam_country_Ghana', 'steam_country_Gibraltar', 'steam_country_Greece', 'steam_country_Greenland', 'steam_country_Grenada', 'steam_country_Guadeloupe', 'steam_country_Guam', 'steam_country_Guatemala', 'steam_country_Guernsey', 'steam_country_Guinea', 'steam_country_Guyana', 'steam_country_Haiti', 'steam_country_Honduras', 'steam_country_Hong Kong', 'steam_country_Hungary', 'steam_country_Iceland', 'steam_country_India', 'steam_country_Indonesia', 'steam_country_Iran', 'steam_country_Iraq', 'steam_country_Ireland', 'steam_country_Isle of Man', 'steam_country_Israel', 'steam_country_Italy', 'steam_country_Jamaica', 'steam_country_Japan', 'steam_country_Jordan', 'steam_country_Kazakhstan', 'steam_country_Kenya', 'steam_country_Korea, Republic of', 'steam_country_Kuwait', 'steam_country_Kyrgyzstan', "steam_country_Lao People's Democratic Republic", 'steam_country_Latvia', 'steam_country_Lebanon', 'steam_country_Libya', 'steam_country_Liechtenstein', 'steam_country_Lithuania', 'steam_country_Luxembourg', 'steam_country_Macau', 'steam_country_Macedonia, the former Yugoslav Republic of', 'steam_country_Madagascar', 'steam_country_Malawi', 'steam_country_Malaysia', 'steam_country_Maldives', 'steam_country_Malta', 'steam_country_Martinique', 'steam_country_Mauritius', 'steam_country_Mexico', 'steam_country_Moldova, Republic of', 'steam_country_Mongolia', 'steam_country_Montenegro', 'steam_country_Morocco', 'steam_country_Mozambique', 'steam_country_Myanmar', 'steam_country_Namibia', 'steam_country_Nepal', 'steam_country_Netherlands', 'steam_country_New Caledonia', 'steam_country_New Zealand', 'steam_country_Nicaragua', 'steam_country_Nigeria', 'steam_country_Norway', 'steam_country_Oman', 'steam_country_Pakistan', 'steam_country_Palestinian Territory, Occupied', 'steam_country_Panama', 'steam_country_Paraguay', 'steam_country_Peru', 'steam_country_Philippines', 'steam_country_Poland', 'steam_country_Portugal', 'steam_country_Puerto Rico', 'steam_country_Qatar', 'steam_country_Reunion', 'steam_country_Romania', 'steam_country_Russian Federation', 'steam_country_Rwanda', 'steam_country_Saint Lucia', 'steam_country_San Marino', 'steam_country_Saudi Arabia', 'steam_country_Senegal', 'steam_country_Serbia', 'steam_country_Seychelles', 'steam_country_Singapore', 'steam_country_Slovakia', 'steam_country_Slovenia', 'steam_country_South Africa', 'steam_country_Spain', 'steam_country_Sri Lanka', 'steam_country_Suriname', 'steam_country_Sweden', 'steam_country_Switzerland', 'steam_country_Taiwan', 'steam_country_Tajikistan', 'steam_country_Thailand', 'steam_country_Timor-Leste', 'steam_country_Togo', 'steam_country_Tokelau', 'steam_country_Trinidad and Tobago', 'steam_country_Tunisia', 'steam_country_Turkey', 'steam_country_Turkmenistan', 'steam_country_Turks and Caicos Islands', 'steam_country_Tuvalu', 'steam_country_Uganda', 'steam_country_Ukraine', 'steam_country_United Arab Emirates', 'steam_country_United Kingdom', 'steam_country_United States', 'steam_country_Uruguay', 'steam_country_Uzbekistan', 'steam_country_Venezuela', 'steam_country_Viet Nam', 'steam_country_Virgin Islands, U.S.', 'steam_country_Zambia', 'steam_country_Zimbabwe', 'steam_community_ban_0', 'steam_community_ban_1', 'steam_is_limited_0', 'steam_is_limited_1', 'steam_cs2_wingman_rank_id_0', 'steam_cs2_wingman_rank_id_1', 'steam_cs2_wingman_rank_id_2', 'steam_cs2_wingman_rank_id_3', 'steam_cs2_wingman_rank_id_4', 'steam_cs2_wingman_rank_id_5', 'steam_cs2_wingman_rank_id_6', 'steam_cs2_wingman_rank_id_7', 'steam_cs2_wingman_rank_id_8', 'steam_cs2_wingman_rank_id_9', 'steam_cs2_wingman_rank_id_10', 'steam_cs2_wingman_rank_id_11', 'steam_cs2_wingman_rank_id_12', 'steam_cs2_wingman_rank_id_13', 'steam_cs2_wingman_rank_id_14', 'steam_cs2_wingman_rank_id_15', 'steam_cs2_wingman_rank_id_16', 'steam_cs2_wingman_rank_id_17', 'steam_cs2_wingman_rank_id_18', 'steam_cs2_rank_id_0', 'steam_cs2_rank_id_1', 'steam_cs2_rank_id_2', 'steam_cs2_rank_id_3', 'steam_cs2_rank_id_4', 'steam_cs2_rank_id_5', 'steam_cs2_rank_id_6', 'steam_cs2_rank_id_7', 'steam_cs2_rank_id_8', 'steam_cs2_rank_id_9', 'steam_cs2_rank_id_10', 'steam_cs2_rank_id_11', 'steam_cs2_rank_id_12', 'steam_cs2_rank_id_13', 'steam_cs2_rank_id_14', 'steam_cs2_rank_id_15', 'steam_cs2_rank_id_16', 'steam_cs2_rank_id_17', 'steam_cs2_rank_id_18', 'steam_cs2_ban_type_0', 'steam_currency_ AED', 'steam_currency_ KD', 'steam_currency_ QR', 'steam_currency_ SR', 'steam_currency_ TL', 'steam_currency_ kr', 'steam_currency_ pуб', 'steam_currency_ €', 'steam_currency_ ₴', 'steam_currency_ ₸', 'steam_currency_$', 'steam_currency_$ USD', 'steam_currency_$ USD USD', 'steam_currency_$U', 'steam_currency_A$ ', 'steam_currency_ARS$ ', 'steam_currency_CDN$ ', 'steam_currency_CHF ', 'steam_currency_CLP$ ', 'steam_currency_COL$ ', 'steam_currency_HK$ ', 'steam_currency_Mex$ ', 'steam_currency_NT$ ', 'steam_currency_NZ$ ', 'steam_currency_P', 'steam_currency_R ', 'steam_currency_R$ ', 'steam_currency_RM', 'steam_currency_Rp ', 'steam_currency_Rp  ', 'steam_currency_S$', 'steam_currency_S/', 'steam_currency_zł', 'steam_currency_£', 'steam_currency_¥ ', 'steam_currency_฿', 'steam_currency_₡', 'steam_currency_₩ ', 'steam_currency_₪', 'steam_currency_₫', 'steam_currency_€', 'steam_currency_₴', 'steam_currency_₸', 'steam_currency_₹ ', 'published_date_is_weekend_0', 'published_date_is_weekend_1', 'update_stat_date_is_weekend_0', 'update_stat_date_is_weekend_1', 'refreshed_date_is_weekend_0', 'refreshed_date_is_weekend_1', 'steam_register_date_is_weekend_0', 'steam_register_date_is_weekend_1', 'steam_last_activity_is_weekend_0', 'steam_last_activity_is_weekend_1', 'steam_cs2_last_activity_is_weekend_0', 'steam_cs2_last_activity_is_weekend_1', 'steam_cs2_ban_date_is_weekend_0', 'steam_cs2_ban_date_is_weekend_1', 'steam_last_transaction_date_is_weekend_0', 'steam_last_transaction_date_is_weekend_1', 'steam_market_ban_end_date_is_weekend_0', 'steam_market_ban_end_date_is_weekend_1', 'steam_cs2_last_launched_is_weekend_0', 'steam_cs2_last_launched_is_weekend_1']

class SingleCategoryModel:
    def __init__(self, category_number):
        """
        Initializes the SingleCategoryModel class with the category number.

        Parameters:
        - category_number: int - The category number for which the model is trained.
        """
        self.category_number = category_number
        self.meta_model = None
        print(f"Initialized SingleCategoryModel for category {self.category_number}.")
        
    def one_hot_encode_and_drop(self, df, columns):
        """
        One-hot-encodes the specified columns in a DataFrame and drops the original columns.
        
        Parameters:
        - df: pd.DataFrame - Input DataFrame.
        - columns: list of str - List of column names to be one-hot-encoded.
        
        Returns:
        - pd.DataFrame - Updated DataFrame with one-hot-encoded columns and original columns dropped.
        """
        # Ensure columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following columns are not in the DataFrame: {missing_columns}")
        
        # Perform one-hot encoding
        encoded_df = pd.get_dummies(df, columns=columns, drop_first=False)
        
        return encoded_df
        
    def load_model(self, fname: str):
        self.meta_model = CatBoostRegressor(
                iterations=60000,
                l2_leaf_reg=2.3,
                use_best_model=True,
                early_stopping_rounds=100,
                posterior_sampling=True,
                grow_policy='SymmetricTree',
                bootstrap_type='Bernoulli',
                random_state=42,
                leaf_estimation_method='Newton',
                score_function='Cosine',
                thread_count=-1).load_model(fname=fname, format='onnx')
        print("Loaded the model from", fname)
    
    def sum_playtime(self, x):
            
            s = 0
            
            if isinstance(x, dict):
                if 'list' in list(x.keys()):
                    for key in x['list']:
                        s += float(x['list'][key]['playtime_forever'])
                
            return s

    def std_playtime(self, x):
        playtimes = []
        std = 0
            
        if isinstance(x, dict):
            if 'list' in list(x.keys()):
                for key in x['list']:
                    playtimes.append(float(x['list'][key]['playtime_forever']))
                std = np.std(playtimes)
        return std

    def preprocess_data(self, df):
        """
        Preprocesses the input dataset.

        Parameters:
        - df: DataFrame.

        Returns:
        - pd.DataFrame - Preprocessed DataFrame.
        """
        df = df.copy()

        # Drop unnecessary columns
        df = df.drop(columns=['steam_cards_count', 'steam_cards_games', 'category_id', 'is_sticky'])

        # Convert timestamp columns to datetime
        date_cols = ['published_date', 'update_stat_date', 'refreshed_date', 'steam_register_date',
                     'steam_last_activity', 'steam_cs2_last_activity', 'steam_cs2_ban_date',
                     'steam_last_transaction_date', 'steam_market_ban_end_date', 'steam_cs2_last_launched']
        for col in date_cols:
            df[col] = df[col].apply(lambda x: datetime.fromtimestamp(x) if x != 0 else np.NaN)

        # Extract time features
        for col in date_cols:
            df = self.extract_time_features(df, col)
        df = df.drop(columns=date_cols)

        # Handle `steam_balance`
        df['steam_currency'] = df['steam_balance'].apply(lambda x: self.remove_numbers_dots_dashes(x))
        df = df.drop(columns=['steam_balance'])

        # Sum columns
        df['inv_value_sum'] = df.filter(like='inv_value').sum(axis=1)
        df['game_count_sum'] = df.filter(like='game_count').sum(axis=1)
        df['level_sum'] = df.filter(like='level').sum(axis=1)
        
        # Additional feature engineering
        df['price_per_view'] = df['price'] / df['view_count']
        
        # steam_full_games handling
        df['total_steam_games'] = df['steam_full_games'].apply(lambda x: x['total'] if 'total' in x != None else -1)
        df['total_playtime'] = df['steam_full_games'].apply(lambda x: self.sum_playtime(x))
        df['std_playtime'] = df['steam_full_games'].apply(lambda x: self.std_playtime(x))
        
        df = df.drop(columns=['steam_full_games'])

        # One-hot encode categorical features
        cat_features = ['item_origin', 'extended_guarantee', 'nsb', 'email_type', 'item_domain', 
                        'resale_item_origin', 'steam_country', 'steam_community_ban', 'steam_is_limited',
                'steam_cs2_wingman_rank_id', 'steam_cs2_rank_id', 'steam_cs2_ban_type', 'steam_currency'] + \
                       [col for col in df.columns if 'is_weekend' in col]
                       
        df = self.one_hot_encode_and_drop(df=df, columns=cat_features)
        #df = df.drop(columns=cat_features + ['steam_bans', 'steam_limit_spent', 'steam_has_activated_keys'])
        
        # check if all features are in df
        if len(set(train_columns) - set(df.columns)) > 0:
            for c in list(set(train_columns) - set(df.columns)):
                df[c] = 0
            
        df = df[train_columns]
        
        df.fillna(0, inplace=True)
        
        return df

    @staticmethod
    def extract_time_features(df, col):
        df[col + '_year'] = df[col].dt.year
        df[col + '_month'] = df[col].dt.month
        df[col + '_day'] = df[col].dt.day
        df[col + '_hour'] = df[col].dt.hour
        df[col + '_minute'] = df[col].dt.minute
        df[col + '_second'] = df[col].dt.second
        df[col + '_weekday'] = df[col].dt.weekday
        df[col + '_is_weekend'] = df[col].dt.weekday.isin([5, 6]).astype(int)
        return df

    @staticmethod
    def remove_numbers_dots_dashes(s):
        return re.sub(r'[0-9.,-]', '', s) if isinstance(s, str) else s

    def train(self, df):
        df = self.preprocess_data(df)
        X_train = df.drop(columns=['sold_price'])
        y_train = df['sold_price']
        
        skf = KFold(n_splits=3, random_state=42, shuffle=True)
        models = []
        
        # Initialize the first model
        model = CatBoostRegressor(
            iterations=20000,
            l2_leaf_reg=2.3,
            use_best_model=True,
            early_stopping_rounds=100,
            grow_policy='SymmetricTree',
            bootstrap_type='Bernoulli',
            random_state=42,
            leaf_estimation_method='Newton',
            score_function='Cosine',
            thread_count=-1,
            verbose=0
        )
        
        for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f'TRAINING FOLD {i+1} of 3 total.\n')
            
            if i == 0:
                # Train the first model from scratch
                model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
                        eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]))
            else:
                # Continue training from the previous model
                model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
                        eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
                        init_model=model)
            
            # Save the current model state
            models.append(model.copy())
            
            # Predictions for validation
            preds = model.predict(X_train.iloc[val_idx])
            pearson = self.pearson_correlation_preds_yval(preds, y_train.iloc[val_idx])
            #models_weights.append(pearson)
        
        # Normalize model weights and create meta-model
        #models_weights = self.normalize_weights(models_weights)
        self.meta_model = model  # Store all models for ensemble prediction
        print("Training complete.")
        
        return self.meta_model

    @staticmethod
    def normalize_weights(weights):
        total = sum(weights)
        return [w / total for w in weights]

    def validate(self, valid_df, save_plot_path="pearson_vs_samples.png"):
        """
        Validates the model on a validation set.

        Parameters:
        - valid_df: pd.DataFrame - DataFrame of base structure on which we validate the model.
        - save_plot_path: str - Path to save the Pearson correlation plot.

        Returns:
        - dict - Regression metrics and Pearson correlations.
        """
        # Preprocess validation data
        valid_df = self.preprocess_data(valid_df)
        X_val = valid_df.drop(columns=['sold_price'])
        y_val = valid_df['sold_price']
        
        # Predict using the model
        preds = self.meta_model.predict(X_val)
        
        # Calculate regression metrics
        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)
        pearson_corr_full = self.pearson_correlation_preds_yval(preds, y_val)
        
        # Pearson correlations for subsets
        sample_pearsons = {}
        for size in [100, 1000, 10000]:
            if len(y_val) >= size:
                indices = np.random.choice(len(y_val), size=size, replace=False)
                sampled_preds = preds[indices]
                sampled_y_val = y_val.iloc[indices]
                sample_pearsons[size] = self.pearson_correlation_preds_yval(sampled_preds, sampled_y_val)
            else:
                sample_pearsons[size] = None
        
        # Print and log metrics
        print("\nRegression Metrics:\n")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Pearson Correlation (Full Dataset): {pearson_corr_full:.4f}")
        for size, corr in sample_pearsons.items():
            if corr is not None:
                print(f"Pearson Correlation (Sample Size {size}): {corr:.4f}")
            else:
                print(f"Pearson Correlation (Sample Size {size}): Not enough rows in validation set")
        
        # Generate Pearson correlation vs. number of samples plot
        self._plot_pearson_correlation(preds, y_val, save_path=save_plot_path)
        
        # Return metrics as a dictionary
        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "pearson_correlation_full": pearson_corr_full,
            "sample_pearsons": sample_pearsons
        }
        
    def finetune(self, df, use_crossval=True):
        """Fine-tunes the existing CatBoost model with new data.

        Args:
            df (DataFrame): DataFrame containing the new data to fine-tune on.
            use_crossval (bool, optional): Whether to use cross-validation during fine-tuning. Defaults to True.
        """
        # Preprocess the new data
        df = self.preprocess_data(df)
        
        # Separate features and target variable
        X_train = df.drop(columns=['sold_price'])
        y_train = df['sold_price']
        
        # Initialize a new CatBoostRegressor with the same parameters as the meta_model
        new_model = CatBoostRegressor(
            iterations=60000,
            l2_leaf_reg=2.3,
            use_best_model=True,
            early_stopping_rounds=100,
            grow_policy='SymmetricTree',
            bootstrap_type='Bernoulli',
            random_state=42,
            leaf_estimation_method='Newton',
            score_function='Cosine',
            thread_count=-1
        )
        
        # Check if meta_model is available and is of the correct type
        if not hasattr(self, 'meta_model') or not isinstance(self.meta_model, CatBoostRegressor):
            raise ValueError("meta_model is not available or not a CatBoostRegressor instance.")
        
        try:
            if use_crossval:
                # Split the data into training and validation sets
                from sklearn.model_selection import train_test_split
                X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, random_state=42)
                
                # Fine-tune the model with cross-validation
                new_model.fit(X_train_split, y_train_split, 
                            eval_set=(X_test_split, y_test_split), 
                            verbose=0, 
                            init_model=self.meta_model)
            else:
                # Fine-tune the model without cross-validation
                new_model.fit(X_train, y_train, 
                            verbose=0, 
                            init_model=self.meta_model)
            
            # Update the meta_model with the fine-tuned model
            self.meta_model = new_model
            print("Finetuning complete.")
        except Exception as e:
            print(f"An error occurred during finetuning: {e}")

    @staticmethod
    def pearson_correlation_preds_yval(preds, y_val):
        preds, y_val = np.array(preds), np.array(y_val)
        return np.corrcoef(preds, y_val)[0, 1]

    def export(self, output_path_onnx):
        """
        Exports the trained model to an ONNX file.

        Parameters:
        - output_path: str - Path to save the ONNX model.
        """
        self.meta_model.save_model(
            fname=output_path_onnx,
            format="onnx",
            export_parameters={
                'onnx_domain': 'ai.catboost',
                'onnx_model_version': 1,
                'onnx_doc_string': f'Model for category {self.category_number}',
                'onnx_graph_name': f'Category {self.category_number} CatBoost Regressor'
            }
        )

        print(f"Model exported to {output_path_onnx}")

    def _plot_pearson_correlation(self, preds, y_val, save_path="pearson_vs_samples.png"):
        """
        Generates a plot for Pearson correlation vs. number of samples with trend and LOWESS.

        Parameters:
        - preds: np.array - Predicted values.
        - y_val: np.array - True values.
        - save_path: str - Path to save the plot.
        """
        # Prepare data for plotting
        sample_sizes = np.linspace(10, len(preds), 100, dtype=int)
        correlations = []

        for size in sample_sizes:
            indices = np.random.choice(len(preds), size=size, replace=False)
            sampled_preds = preds[indices]
            sampled_y_val = y_val.iloc[indices]
            correlations.append(self.pearson_correlation_preds_yval(sampled_preds, sampled_y_val))
        
        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            "Sample Size": sample_sizes,
            "Pearson Correlation": correlations
        })
        
        # Calculate LOWESS trend
        lowess_result = lowess(plot_df["Pearson Correlation"], plot_df["Sample Size"], frac=0.3)
        lowess_x, lowess_y = zip(*lowess_result)
        
        # Plot with seaborn
        sns.set(style="whitegrid", context="talk")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="Sample Size", y="Pearson Correlation", data=plot_df, s=30, alpha=0.7, color="blue", label="Correlation")
        sns.lineplot(x="Sample Size", y="Pearson Correlation", data=plot_df, color="orange", linewidth=2, label="Trend")
        plt.plot(lowess_x, lowess_y, color="red", linestyle="--", linewidth=2, label="LOWESS Curve")
        
        # Add labels and title
        plt.title(f"Pearson Correlation vs. Sample Size (Category {self.category_number})", fontsize=16)
        plt.xlabel("Sample Size", fontsize=14)
        plt.ylabel("Pearson Correlation Coefficient", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_path)
        plt.close()
        print(f"Pearson correlation plot saved to {save_path}")