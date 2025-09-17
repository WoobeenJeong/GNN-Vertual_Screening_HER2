
"""
Command Example:
    python 11_apply_pains_filter.py

Description:
    이 스크립트는 입력 CSV 파일('smiles', 'SA', 'QED' 열 포함)을 읽어,
    SA < 3, QED > 0.7 기준을 만족하는 화합물에 대해 PAINS 필터를 적용합니다.
    결과로 'PAINS'와 'PAINS_why' 열을 추가하여 새로운 CSV 파일로 저장합니다.
"""

    from rdkit import Chem
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    from tqdm import tqdm
    use_tqdm = True

    parser = argparse.ArgumentParser(description="SMILES 데이터에 SA, QED 및 PAINS 필터를 적용합니다.")
    parser.add_argument('--input_csv', type=str, default='/home/ssm-user/project/scores/results/top10000_enamine_ligand_qed_07_sa_3.csv', help="입력 CSV 파일 경로. 'smiles', 'SA', 'QED' 열이 있어야 합니다.")
    parser.add_argument('--output_csv', type=str, default='/home/ssm-user/project/scores/results/top10000_qed_07_sa_3_pains_filtered.csv', help="필터링된 결과를 저장할 CSV 파일 경로.")
    parser.add_argument('--smiles_column', type=str, default='smiles', help="SMILES 데이터가 포함된 열의 이름.")
    parser.add_argument('--sa_threshold', type=float, default=3.0, help="SA (Synthetic Accessibility) 점수 임계값 (이 값 미만).")
    parser.add_argument('--qed_threshold', type=float, default=0.7, help="QED (Quantitative Estimate of Drug-likeness) 점수 임계값 (이 값 초과).")
 
    args = parser.parse_args()

    print(f"입력 파일: {args.input_csv}")
    print(f"출력 파일: {args.output_csv}")

    # PAINS 카탈로그 초기화
    print("PAINS 필터 카탈로그를 초기화합니다...")
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains_catalog = FilterCatalog(params)
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"[error] 입력 파일을 찾을 수 없습니다: {args.input_csv}")
        return

    required_cols = [args.smiles_column, 'SA', 'QED']
    if not all(col in df.columns for col in required_cols):
        print(f"[error] 입력 CSV에 다음 열들이 모두 포함되어야 합니다: {required_cols}")
        return

    def get_pains_info(smiles):
        """주어진 SMILES의 PAINS 정보를 확인하여 (상태, 이유) 튜플을 반환합니다."""
        if not isinstance(smiles, str) or pd.isna(smiles):
            return (0, 'Missing or invalid SMILES string')

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (0, 'Invalid SMILES')
        
        matches = pains_catalog.GetMatches(mol)
        if matches:
            first_match_description = matches[0].GetDescription()
            return (0, f'PAINS: {first_match_description}')
        else:
            return (1, 'OK')

    total_mols = len(df)
    print(f"총 {total_mols}개의 화합물을 처리합니다...")

    # 결과 열 초기화
    # PAINS: 999 (기준 미달), 0 (PAINS 해당), 1 (PAINS 미해당)
    df['PAINS'] = 999
    df['PAINS_why'] = 'Did not meet SA/QED criteria'

    # SA 및 QED 기준을 만족하는 행에 대한 마스크 생성
    mask = (df['SA'] < args.sa_threshold) & (df['QED'] > args.qed_threshold)
    passed_prefilter_count = mask.sum()
    print(f"SA < {args.sa_threshold} 및 QED > {args.qed_threshold} 기준을 만족하는 화합물: {passed_prefilter_count}개")
    
    # 기준을 만족하는 화합물에 대해서만 PAINS 필터 적용
    if passed_prefilter_count > 0:
        smiles_to_check = df.loc[mask, args.smiles_column]
        if use_tqdm:
            pains_results = smiles_to_check.progress_apply(get_pains_info)
        else:
            pains_results = smiles_to_check.apply(get_pains_info)
        
        # 결과를 데이터프레임의 해당 위치에 다시 할당
        df.loc[mask, ['PAINS', 'PAINS_why']] = pd.DataFrame(pains_results.tolist(), index=df.loc[mask].index)

    # 최종 결과 요약
    skipped_prefilter = (df['PAINS'] == 999).sum()
    pains_compounds = (df.loc[mask, 'PAINS'] == 0).sum()
    passed_all_filters = (df['PAINS'] == 1).sum()

    print("\n--- 필터링 결과 ---")
    print(f"총 화합물 수: {total_mols}")
    print(f"PAINS 구조를 포함하거나 유효하지 않은 화합물 수: {num_pains}")
    print(f"필터링 후 남은 화합물 수: {num_kept}")
    print(f"SA/QED 기준 미충족: {skipped_prefilter}")
    print(f"SA/QED 기준 충족: {passed_prefilter_count}")
    print(f"  - PAINS 구조 포함 / 유효하지 않은 SMILES: {pains_compounds}")
    print(f"  - 최종 통과 (SA/QED/PAINS 만족): {passed_all_filters}")
    print("--------------------")

    # 결과 저장
    df.to_csv(args.output_csv, index=False)
    print(f"\n필터링된 데이터가 '{args.output_csv}' 파일에 저장되었습니다.")


if __name__ == "__main__":
    main()

