# 法的評価基準エンティティ設計

## 📋 文書情報

| 項目       | 内容                         |
| ---------- | ---------------------------- |
| 文書名     | 法的評価基準エンティティ設計 |
| バージョン | 1.0                          |
| 作成日     | 2025 年 9 月 28 日           |
| 作成者     | AI 開発チーム                |
| 承認者     | 技術責任者                   |
| ステータス | 草案                         |

## 🎯 概要

法的評価基準エンティティは、法的条文に基づく 6 つの不開示事由の詳細な評価基準を管理するドメインエンティティです。段階的評価プロセス（3-4 段階）と専門的判断ロジックを実装し、法的根拠に基づく厳密な評価を提供します。各不開示事由について詳細な評価プロンプト、豊富な具体例、および段階的評価手順を提供し、専門家レベルの判断を支援します。

## 🏗️ エンティティ設計

### 1. 基本構造

```python
@dataclass
class LegalEvaluationCriteria:
    """
    法的評価基準エンティティ

    法的条文に基づく6つの不開示事由の詳細な評価基準を管理。
    段階的評価プロセス（3-4段階）と専門的判断ロジックを実装。
    詳細な評価プロンプト、豊富な具体例、法的根拠を提供。
    """

    article: LegalArticle
    name: str
    description: str
    legal_basis: str
    weight: float
    evaluation_prompt: str
    evaluation_guidelines: Dict[str, Any]
    scoring_interpretation: Dict[str, str]
    evaluation_steps: List[str]
    examples: Dict[str, Dict[str, Any]]
    reference_text: Dict[str, str]

    def evaluate_information(self, information: str, context: Dict[str, Any] = None) -> "LegalEvaluationResult":
        """
        行政情報を評価して開示判断を行う

        Args:
            information: 評価対象の行政情報
            context: 追加のコンテキスト情報

        Returns:
            LegalEvaluationResult: 法的評価結果
        """
        # 段階的評価プロセスの実装
        # 1. 情報の分類と特定
        # 2. 不開示事由の該当性評価
        # 3. 公益との比較衡量
        # 4. 最終的な開示判断
        pass

    def get_decision_from_score(self, score: float) -> DisclosureDecision:
        """スコアから開示判断を決定"""
        if score >= 4.0:
            return DisclosureDecision.DISCLOSE
        elif score >= 3.0:
            return DisclosureDecision.CONSIDER_DISCLOSURE
        else:
            return DisclosureDecision.NON_DISCLOSURE
```

### 2. 属性説明

| 属性名                     | 型              | 説明                         |
| -------------------------- | --------------- | ---------------------------- |
| **article**                | LegalArticle    | 法的条文（第 1 号〜第 6 号） |
| **name**                   | str             | 不開示事由の名称             |
| **description**            | str             | 不開示事由の詳細説明         |
| **legal_basis**            | str             | 法的根拠条文                 |
| **weight**                 | float           | 重み付け（0.0-1.0）          |
| **evaluation_prompt**      | str             | 詳細な評価プロンプト         |
| **evaluation_guidelines**  | Dict[str, Any]  | 評価ガイドライン（詳細）     |
| **scoring_interpretation** | Dict[str, str]  | スコア解釈基準（1-5）        |
| **evaluation_steps**       | List[str]       | 段階的評価ステップ           |
| **examples**               | Dict[str, Dict] | 豊富な具体例（スコア別）     |
| **reference_text**         | Dict[str, str]  | 参考条文テキスト（詳細）     |

## 📋 法的条文

### 1. 第 1 号：個人情報保護

````python
PERSONAL_INFO_CRITERIA = LegalEvaluationCriteria(
    article=LegalArticle.PERSONAL_INFO,
    name="個人情報保護",
    description="個人に関する情報（事業を営む個人の当該事業に関する情報を除く。）であって、当該情報に含まれる氏名、生年月日その他の記述等により特定の個人を識別することができるもの又は特定の個人を識別することはできないが、公にすることにより、なお個人の権利利益を害するおそれがあるもの",
    legal_basis="情報公開法第5条第1号 - 個人に関する情報（事業を営む個人の当該事業に関する情報を除く。）であって、当該情報に含まれる氏名、生年月日その他の記述等により特定の個人を識別することができるもの又は特定の個人を識別することはできないが、公にすることにより、なお個人の権利利益を害するおそれがあるもの",
    weight=0.2,
    evaluation_prompt="""
    あなたは情報公開法の専門家として、個人情報保護に関する不開示判断を行います。

    ## 評価対象
    情報公開法第5条第1号に基づく個人情報の不開示事由の該当性

    ## 評価手順
    以下の3段階で体系的に評価してください：

    ### 段階1: 基本要件の確認
    以下のすべてを確認し、該当する場合は「YES」、該当しない場合は「NO」で回答：
    1. 個人に関する情報である（事業を営む個人の当該事業に関する情報は除く）
    2. 氏名、生年月日その他の記述等により特定の個人を識別できる
    3. または、特定の個人を識別できないが、公にすることで個人の権利利益を害するおそれがある

    ### 段階2: 開示例外の確認
    以下のいずれかに該当する場合は「YES」、該当しない場合は「NO」で回答：
    1. 法令の規定により又は慣行として公にされ、又は公にすることが予定されている情報
    2. 人の生命、健康、生活又は財産を保護するため、公にすることが必要であると認められる情報
    3. 公務員等の職務の遂行に係る情報であるときは、当該公務員等の職及び当該職務遂行の内容に係る部分

    ### 段階3: 最終判断
    - 段階1で「YES」かつ段階2で「NO」→ 不開示
    - 段階1で「YES」かつ段階2で「YES」→ 開示を検討
    - 段階1で「NO」→ 開示

    ## 重要な注意事項
    ⚠️ 個人のプライバシーと権利利益を慎重に評価し、開示例外の要件を厳格に適用してください。

    ## 出力形式
    各段階の結果と最終判断を以下の形式で出力してください：
    ```
    段階1: [YES/NO] - [理由]
    段階2: [YES/NO] - [理由]
    最終判断: [不開示/開示を検討/開示] - [根拠]
    ```
    """,
    evaluation_guidelines={
        "overall_assessment": "個人情報保護の観点から慎重に評価し、プライバシー権を最優先に考慮する。スコア1-2の場合は不開示の可能性が高い。",
        "decision_process": [
            "個人情報の特定可能性の確認",
            "開示例外要件の厳格な適用",
            "1-5スケールでのスコア付け",
            "プライバシー権と公益のバランス考慮",
            "部分開示の可能性の検討"
        ]
    },
    scoring_interpretation={
        "1": "強く不開示 - 明確に個人情報保護規定に該当",
        "2": "不開示の可能性が高い - 個人情報保護規定に該当する可能性が高い",
        "3": "不明確 - 慎重な法的検討が必要",
        "4": "開示の可能性が高い - 適切な保護措置と共に開示可能",
        "5": "明確に開示 - 個人情報に該当しないか開示例外に該当"
    },
    evaluation_steps=[
        "個人に関する情報（事業を営む個人の当該事業に関する情報を除く。）か",
        "特定の個人を識別することができるものか",
        "特定の個人を識別することはできないが、公にすることにより、なお個人の権利利益を害するおそれがあるものか",
        "法令の規定により又は慣行として公にされ、又は公にすることが予定されている情報か",
        "人の生命、健康、生活又は財産を保護するため、公にすることが必要であると認められる情報か",
        "公務員等の職務の遂行に係る情報であるときは、当該公務員等の職及び当該職務遂行の内容に係る部分か"
    ],
    examples={
        "1": {
            "severity_level": "最高",
            "description": "高度に機密性の高い個人情報",
            "specific_cases": [
                "医療記録（診断書、治療履歴、精神科記録）",
                "財務データ（所得証明書、納税記録、銀行口座情報）",
                "家族関係（養子縁組記録、親子関係証明書）",
                "犯罪歴・前科記録",
                "精神保健福祉手帳の記録",
                "生活保護受給記録",
                "児童相談所の記録",
                "DV被害者情報"
            ],
            "evaluation_result": "不開示",
            "reasoning": "個人のプライバシーと権利利益を著しく害するおそれ"
        },
        "5": {
            "severity_level": "なし",
            "description": "個人情報がないか開示例外に該当",
            "specific_cases": [
                "政策文書（個人情報を含まない）",
                "行政手続きの一般的な説明",
                "統計データ（個人を特定できない形）",
                "法令・規則・通達",
                "予算・決算書類（個人情報を含まない部分）",
                "公共事業の概要・計画",
                "環境影響評価書（個人情報を含まない部分）",
                "研究報告書（個人情報を含まない部分）"
            ],
            "evaluation_result": "開示",
            "reasoning": "個人情報に該当しないか、開示例外に該当"
        }
    },
    reference_text={
        "source": "情報公開法第5条第1号",
        "content": "個人に関する情報（事業を営む個人の当該事業に関する情報を除く。）であって、当該情報に含まれる氏名、生年月日その他の記述等（文書、図画若しくは電磁的記録に記載され、若しくは記録され、又は音声、動作その他の方法を用いて表された一切の事項をいう。次条第二項において同じ。）により特定の個人を識別することができるもの（他の情報と照合することにより、特定の個人を識別することができることとなるものを含む。）又は特定の個人を識別することはできないが、公にすることにより、なお個人の権利利益を害するおそれがあるもの。ただし、次に掲げる情報を除く。イ　法令の規定により又は慣行として公にされ、又は公にすることが予定されている情報　ロ　人の生命、健康、生活又は財産を保護するため、公にすることが必要であると認められる情報　ハ　当該個人が公務員等（国家公務員法（昭和二十二年法律第百二十号）第二条第一項に規定する国家公務員（独立行政法人通則法（平成十一年法律第百三号）第二条第四項に規定する行政執行法人の役員及び職員を除く。）、独立行政法人等（独立行政法人等の保有する情報の公開に関する法律（平成十三年法律第百四十号。以下「独立行政法人等情報公開法」という。）第二条第一項に規定する独立行政法人等をいう。以下同じ。）の役員及び職員、地方公務員法（昭和二十五年法律第二百六十一号）第二条に規定する地方公務員並びに地方独立行政法人（地方独立行政法人法（平成十五年法律第百十八号）第二条第一項に規定する地方独立行政法人をいう。以下同じ。）の役員及び職員をいう。）である場合において、当該情報がその職務の遂行に係る情報であるときは、当該情報のうち、当該公務員等の職及び当該職務遂行の内容に係る部分"
    }
)
````

### 2. 第 2 号：法人等情報保護

```python
CORPORATE_INFO_CRITERIA = LegalEvaluationCriteria(
    article=LegalArticle.CORPORATE_INFO,
    name="法人等情報保護",
    description="法人等の正当な利益を害する情報の保護",
    legal_basis="情報公開法第5条第2号",
    evaluation_guidelines="""
    法人等情報の該当性を以下の観点から評価：
    1. 法人等の正当な利益に属する情報か
    2. 公開による不利益の程度
    3. 競争上の不利益の可能性
    """,
    evaluation_steps=[
        "法人等の正当な利益の確認",
        "競争上の不利益の評価",
        "公開による影響の評価",
        "公益との比較衡量"
    ],
    scoring_interpretation={
        1: "明確に法人等情報、強く保護必要",
        2: "法人等情報の可能性高、保護必要",
        3: "法人等情報の可能性中程度、要検討",
        4: "法人等情報の可能性低、開示可能",
        5: "法人等情報に該当せず、開示可能"
    },
    examples=[
        {
            "information": "当社の新商品開発計画書",
            "expected_score": 1,
            "reasoning": "競争上の不利益を生じる可能性が高い"
        }
    ],
    reference_text="情報公開法第5条第2号の条文"
)
```

### 3. 第 3 号：国家安全保障

```python
NATIONAL_SECURITY_CRITERIA = LegalEvaluationCriteria(
    article=LegalArticle.NATIONAL_SECURITY,
    name="国家安全保障",
    description="国家安全保障に影響する情報の保護",
    legal_basis="情報公開法第5条第3号",
    evaluation_guidelines="""
    国家安全保障情報の該当性を以下の観点から評価：
    1. 国家安全保障に影響する情報か
    2. 公開による安全保障上の不利益
    3. 国際関係への影響
    """,
    evaluation_steps=[
        "安全保障上の重要性の確認",
        "公開による不利益の評価",
        "国際関係への影響評価",
        "公益との比較衡量"
    ],
    scoring_interpretation={
        1: "明確に安全保障情報、絶対保護必要",
        2: "安全保障情報の可能性高、保護必要",
        3: "安全保障情報の可能性中程度、要検討",
        4: "安全保障情報の可能性低、開示可能",
        5: "安全保障情報に該当せず、開示可能"
    },
    examples=[
        {
            "information": "防衛省の装備調達計画",
            "expected_score": 1,
            "reasoning": "国家安全保障に直接影響する情報"
        }
    ],
    reference_text="情報公開法第5条第3号の条文"
)
```

## 🔄 段階的評価プロセス

### 1. 詳細な評価プロセス

```python
def evaluate_information(self, information: str, context: Dict[str, Any] = None) -> "LegalEvaluationResult":
    """行政情報の段階的評価を実行（3-4段階の詳細プロセス）"""

    # ステップ1: 情報の分類と特定
    classification_result = self._classify_information(information, context)

    # ステップ2: 段階的評価プロンプトの適用
    evaluation_result = self._apply_evaluation_prompt(information, classification_result)

    # ステップ3: 不開示事由の該当性評価（段階的）
    applicability_result = self._evaluate_applicability_staged(information, evaluation_result)

    # ステップ4: 開示例外要件の確認
    exception_result = self._check_disclosure_exceptions(information, applicability_result)

    # ステップ5: 公益との比較衡量
    public_interest_result = self._evaluate_public_interest(information, exception_result)

    # ステップ6: 最終的な開示判断
    final_decision = self._determine_final_decision(
        classification_result,
        evaluation_result,
        applicability_result,
        exception_result,
        public_interest_result
    )

    return LegalEvaluationResult(
        article=self.article,
        decision=final_decision,
        reasoning=self._generate_detailed_reasoning(
            classification_result,
            evaluation_result,
            applicability_result,
            exception_result,
            public_interest_result,
            final_decision
        ),
        confidence_score=self._calculate_confidence(
            classification_result,
            evaluation_result,
            applicability_result,
            exception_result,
            public_interest_result
        ),
        legal_basis=self._generate_legal_basis(final_decision),
        partial_disclosure_options=self._consider_partial_disclosure(
            information, final_decision
        )
    )
```

### 2. 詳細な評価プロンプトの適用

```python
def _apply_evaluation_prompt(self, information: str, classification: ClassificationResult) -> EvaluationResult:
    """詳細な評価プロンプトを適用して段階的評価を実行"""

    # 評価プロンプトの適用
    evaluation_prompt = f"""
    {self.evaluation_prompt}

    評価対象情報:
    {information}

    分類結果:
    {classification.reasoning}

    評価手順に従って段階的に評価してください。
    """

    # LLMを使用した段階的評価実行
    response = await self._llm_provider.generate_response(evaluation_prompt)
    return self._parse_evaluation_response(response)

def _evaluate_applicability_staged(self, information: str, evaluation: EvaluationResult) -> ApplicabilityResult:
    """段階的評価手順に基づく該当性評価"""

    if evaluation.stage1_result == "NO":
        return ApplicabilityResult(
            applicable=False,
            score=5.0,
            reasoning="段階1で該当しないと判断されたため、開示可能"
        )

    # 段階2の評価
    if evaluation.stage2_result == "YES":
        return ApplicabilityResult(
            applicable=True,
            score=1.0,
            reasoning="段階1で該当し、段階2で開示例外に該当しないため、不開示"
        )
    elif evaluation.stage2_result == "NO":
        return ApplicabilityResult(
            applicable=False,
            score=4.0,
            reasoning="段階1で該当するが、段階2で開示例外に該当するため、開示を検討"
        )

    return ApplicabilityResult(
        applicable=True,
        score=2.0,
        reasoning="段階的評価により慎重な検討が必要"
    )
```

### 3. 該当性評価

```python
def _evaluate_applicability(self, information: str, classification: ClassificationResult) -> ApplicabilityResult:
    """不開示事由の該当性評価"""

    if classification.category == "該当しない":
        return ApplicabilityResult(
            applicable=False,
            score=5.0,
            reasoning="該当しないと分類されたため、開示可能"
        )

    applicability_prompt = f"""
    {self.name}の該当性を詳細に評価してください。

    情報: {information}
    分類結果: {classification.reasoning}

    評価ステップ:
    {chr(10).join(f"{i+1}. {step}" for i, step in enumerate(self.evaluation_steps))}

    スコア基準:
    {self._format_scoring_interpretation()}

    以下の形式で回答してください:
    スコア: [1-5の数値]
    理由: [詳細な評価理由]
    信頼度: [0.0-1.0]
    """

    response = await self._llm_provider.generate_response(applicability_prompt)
    return self._parse_applicability_response(response)
```

## 🧪 テスト設計

### 1. 単体テスト

```python
class TestLegalEvaluationCriteria:
    """法的評価基準のテスト"""

    @pytest.fixture
    def personal_info_criteria(self):
        """個人情報保護基準"""
        return LegalEvaluationCriteria(
            article=LegalArticle.PERSONAL_INFO,
            name="個人情報保護",
            description="個人のプライバシーに属する情報の保護",
            legal_basis="情報公開法第5条第1号",
            evaluation_guidelines="個人情報の該当性を評価",
            evaluation_steps=[
                "個人特定可能性の確認",
                "プライバシー侵害の程度評価"
            ],
            scoring_interpretation={
                1: "明確に個人情報、強く保護必要",
                2: "個人情報の可能性高、保護必要",
                3: "個人情報の可能性中程度、要検討",
                4: "個人情報の可能性低、開示可能",
                5: "個人情報に該当せず、開示可能"
            },
            examples=[],
            reference_text="情報公開法第5条第1号の条文"
        )

    def test_get_decision_from_score_disclose(self, personal_info_criteria):
        """スコア4.5の場合に開示判断が返されることをテスト"""
        decision = personal_info_criteria.get_decision_from_score(4.5)
        assert decision == DisclosureDecision.DISCLOSE

    def test_get_decision_from_score_consider(self, personal_info_criteria):
        """スコア3.0の場合に開示を検討判断が返されることをテスト"""
        decision = personal_info_criteria.get_decision_from_score(3.0)
        assert decision == DisclosureDecision.CONSIDER_DISCLOSURE

    def test_get_decision_from_score_non_disclosure(self, personal_info_criteria):
        """スコア2.0の場合に不開示判断が返されることをテスト"""
        decision = personal_info_criteria.get_decision_from_score(2.0)
        assert decision == DisclosureDecision.NON_DISCLOSURE

    def test_evaluate_information_with_personal_data(self, personal_info_criteria):
        """個人情報を含む情報の評価テスト"""
        information = "申請者: 田中太郎, 住所: 東京都新宿区1-1-1"

        # モックを使用した評価テスト
        # 実際の実装では、評価ロジックをテスト
        result = personal_info_criteria.evaluate_information(information)

        assert result is not None
        assert result.article == LegalArticle.PERSONAL_INFO
        assert result.decision in [DisclosureDecision.DISCLOSE,
                                 DisclosureDecision.CONSIDER_DISCLOSURE,
                                 DisclosureDecision.NON_DISCLOSURE]
```

### 2. 統合テスト

```python
class TestLegalEvaluationIntegration:
    """法的評価の統合テスト"""

    @pytest.fixture
    async def legal_evaluation_engine(self):
        """法的評価エンジン"""
        from disclosure_evaluator.domain.services.legal_evaluation_engine import LegalEvaluationEngine

        # モックプロバイダー
        mock_provider = Mock()
        mock_provider.generate_response.return_value = Mock(
            content="分析結果: 個人が特定可能\n判断: 該当する\n根拠: 氏名と住所の組み合わせ"
        )

        # モックリポジトリ
        mock_repository = Mock()
        mock_repository.get_by_article.return_value = self._create_mock_criteria()

        return LegalEvaluationEngine(mock_provider, mock_repository)

    def _create_mock_criteria(self):
        """モック評価基準を作成"""
        return LegalEvaluationCriteria(
            article=LegalArticle.PERSONAL_INFO,
            name="個人情報保護",
            description="個人のプライバシーに属する情報の保護",
            legal_basis="情報公開法第5条第1号",
            evaluation_guidelines="個人情報の該当性を評価",
            evaluation_steps=["個人特定可能性の確認"],
            scoring_interpretation={1: "明確に個人情報", 5: "個人情報に該当せず"},
            examples=[],
            reference_text="テスト用条文"
        )

    async def test_personal_information_evaluation(self, legal_evaluation_engine):
        """個人情報評価の統合テスト"""
        information = "申請者: 田中太郎, 住所: 東京都新宿区1-1-1, 電話: 03-1234-5678"

        result = await legal_evaluation_engine.evaluate_information(information)

        assert result is not None
        assert result.disclosure_decision is not None
        assert len(result.article_scores) > 0
        assert result.legal_reasoning is not None
```

## 🔧 実装ガイドライン

### 1. 法的根拠の正確性

- **条文参照**: 情報公開法の正確な条文番号と内容を参照
- **判例考慮**: 関連する判例や解釈を考慮
- **更新対応**: 法改正への対応
- **専門機関との協議**: 必要に応じて関係省庁や専門機関との協議

### 2. 段階的評価プロセスの実装

- **3-4 段階評価**: 各不開示事由に応じた段階的評価手順
- **詳細な評価プロンプト**: 専門家レベルの判断を支援するプロンプト
- **YES/NO 形式**: 明確な判断基準の適用
- **根拠明示**: 各段階での判断根拠の明確な記録

### 3. 豊富な具体例の活用

- **スコア別事例**: 各スコアレベル（1-5）に対応する具体例
- **重要度別分類**: 最高・高・中・低・なしの重要度分類
- **実践的ケース**: 実際の行政情報に基づく事例

### 4. 品質保証の強化

- **段階的評価手順の遵守**: 各段階の適切な実行
- **開示例外要件の厳格な適用**: 法的要件の正確な適用
- **部分開示の検討**: 適切な部分開示の可能性検討
- **専門機関との連携**: 必要に応じて専門機関との協議

## 📊 パフォーマンス考慮事項

### 1. 評価時間

- **段階的評価**: 各段階の処理時間最適化
- **並列処理**: 複数基準の並列評価
- **キャッシュ**: 類似ケースの結果キャッシュ

### 2. 精度向上

- **学習機能**: 過去の判断結果からの学習
- **フィードバック**: 専門家による判断のフィードバック
- **改善**: 継続的な評価精度の改善

## 🔄 将来の拡張性

### 1. 新しい条文対応

- **法改正**: 情報公開法の改正への対応
- **新基準**: 新しい評価基準の追加
- **国際基準**: 国際的な情報公開基準への対応

### 2. 高度な評価機能

- **機械学習**: ML 技術を活用した評価精度向上
- **自然言語処理**: より高度な NLP 技術の活用
- **専門知識**: 法務専門知識の組み込み

---

_この法的評価基準エンティティ設計により、Disclosure Evaluator は法的根拠に基づく正確で透明性の高い評価システムを実現します。_
