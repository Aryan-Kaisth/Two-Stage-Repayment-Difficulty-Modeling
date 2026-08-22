from typing import Optional
import pandera.pandas as pa
from pandera.typing import Series


# Base Schema
class BaseApplicationSchema(pa.DataFrameModel):
    # Identifiers
    sk_id_curr: Series[pa.Int64] = pa.Field(unique=True, nullable=False, ge=0)

    # Categoricals / Strings (16 columns)
    name_contract_type: Series[pa.String] = pa.Field(nullable=False)
    code_gender: Series[pa.String] = pa.Field(nullable=False)
    flag_own_car: Series[pa.String] = pa.Field(nullable=False)
    flag_own_realty: Series[pa.String] = pa.Field(nullable=False)
    name_type_suite: Optional[Series[pa.String]] = pa.Field(nullable=True)
    name_income_type: Series[pa.String] = pa.Field(nullable=False)
    name_education_type: Series[pa.String] = pa.Field(nullable=False)
    name_family_status: Series[pa.String] = pa.Field(nullable=False)
    name_housing_type: Series[pa.String] = pa.Field(nullable=False)
    occupation_type: Optional[Series[pa.String]] = pa.Field(nullable=True)
    weekday_appr_process_start: Series[pa.String] = pa.Field(nullable=False)
    organization_type: Optional[Series[pa.String]] = pa.Field(nullable=True)
    fondkapremont_mode: Optional[Series[pa.String]] = pa.Field(nullable=True)
    housetype_mode: Optional[Series[pa.String]] = pa.Field(nullable=True)
    wallsmaterial_mode: Optional[Series[pa.String]] = pa.Field(nullable=True)
    emergencystate_mode: Optional[Series[pa.String]] = pa.Field(nullable=True)

    # Core Numerics & Personal Info
    cnt_children: Series[pa.Int64] = pa.Field(ge=0, nullable=False)
    amt_income_total: Series[pa.Float64] = pa.Field(gt=0, nullable=False)
    amt_credit: Series[pa.Float64] = pa.Field(gt=0, nullable=False)
    amt_annuity: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    amt_goods_price: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    region_population_relative: Series[pa.Float64] = pa.Field(nullable=False)
    days_birth: Series[pa.Int64] = pa.Field(nullable=False)
    days_employed: Series[pa.Int64] = pa.Field(nullable=False)
    days_registration: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    days_id_publish: Series[pa.Int64] = pa.Field(nullable=False)
    own_car_age: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    cnt_fam_members: Optional[Series[pa.Float64]] = pa.Field(nullable=True)

    # Binary Flags
    flag_mobil: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_emp_phone: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_work_phone: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_cont_mobile: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_phone: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_email: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)

    # Region / City Ratings & Process Hours
    region_rating_client: Series[pa.Int64] = pa.Field(nullable=False)
    region_rating_client_w_city: Series[pa.Int64] = pa.Field(nullable=False)
    hour_appr_process_start: Series[pa.Int64] = pa.Field(ge=0, le=23, nullable=False)

    # Regional & City Mismatch Flags
    reg_region_not_live_region: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    reg_region_not_work_region: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    live_region_not_work_region: Series[pa.Int64] = pa.Field(
        isin=[0, 1], nullable=False
    )
    reg_city_not_live_city: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    reg_city_not_work_city: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    live_city_not_work_city: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)

    # Normalized External Sources
    ext_source_1: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    ext_source_2: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    ext_source_3: Optional[Series[pa.Float64]] = pa.Field(nullable=True)

    # Building Characteristics - Normalized Averages (_avg)
    apartments_avg: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    basementarea_avg: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    years_beginexpluatation_avg: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    years_build_avg: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    commonarea_avg: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    elevators_avg: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    entrances_avg: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    floorsmax_avg: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    floorsmin_avg: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    landarea_avg: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    livingapartments_avg: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    livingarea_avg: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    nonlivingapartments_avg: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    nonlivingarea_avg: Optional[Series[pa.Float64]] = pa.Field(nullable=True)

    # Building Characteristics - Mode (_mode)
    apartments_mode: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    basementarea_mode: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    years_beginexpluatation_mode: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    years_build_mode: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    commonarea_mode: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    elevators_mode: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    entrances_mode: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    floorsmax_mode: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    floorsmin_mode: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    landarea_mode: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    livingapartments_mode: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    livingarea_mode: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    nonlivingapartments_mode: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    nonlivingarea_mode: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    totalarea_mode: Optional[Series[pa.Float64]] = pa.Field(nullable=True)

    # Building Characteristics - Median (_medi)
    apartments_medi: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    basementarea_medi: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    years_beginexpluatation_medi: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    years_build_medi: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    commonarea_medi: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    elevators_medi: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    entrances_medi: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    floorsmax_medi: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    floorsmin_medi: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    landarea_medi: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    livingapartments_medi: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    livingarea_medi: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    nonlivingapartments_medi: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    nonlivingarea_medi: Optional[Series[pa.Float64]] = pa.Field(nullable=True)

    # Social Circle Observations & Defaults
    obs_30_cnt_social_circle: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    def_30_cnt_social_circle: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    obs_60_cnt_social_circle: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    def_60_cnt_social_circle: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    days_last_phone_change: Optional[Series[pa.Float64]] = pa.Field(nullable=True)

    # Document Flags
    flag_document_2: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_3: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_4: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_5: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_6: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_7: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_8: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_9: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_10: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_11: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_12: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_13: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_14: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_15: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_16: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_17: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_18: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_19: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_20: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)
    flag_document_21: Series[pa.Int64] = pa.Field(isin=[0, 1], nullable=False)

    # Credit Bureau Inquiries
    amt_req_credit_bureau_hour: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    amt_req_credit_bureau_day: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    amt_req_credit_bureau_week: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    amt_req_credit_bureau_mon: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    amt_req_credit_bureau_qrt: Optional[Series[pa.Float64]] = pa.Field(nullable=True)
    amt_req_credit_bureau_year: Optional[Series[pa.Float64]] = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = False


# Train Schema: Inherits all 121 features and adds the mandatory target
class ApplicationTrainSchema(BaseApplicationSchema):
    target: Optional[Series[pa.Int64]] = pa.Field(
        isin=[0, 1],
        nullable=False,
    )


# Test Schema
class ApplicationTestSchema(BaseApplicationSchema):
    target: Optional[Series[pa.Int64]] = pa.Field(
        isin=[0, 1],
        nullable=False,
    )
