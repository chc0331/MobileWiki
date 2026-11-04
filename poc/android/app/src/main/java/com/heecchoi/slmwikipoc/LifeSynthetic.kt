package com.heecchoi.slmwikipoc

data class LifeSyntheticItem(
    val id: String, val orderId: String, val topic: String, val title: String, val text: String, val merchant: String,
)

object LifeSynthetic {
    const val DEV = "life-shopping-dev-v2"
    const val TEST = "life-shopping-test-v2"
    private val merchants = listOf("가람상점", "나무상점", "다온상점", "라온상점")
    private val devProducts = listOf("여행가방", "텀블러", "운동화", "우산", "책상등", "수건", "노트", "냄비", "의자", "물병",
        "모자", "충전기", "장갑", "이어폰", "컵", "이불", "가방", "식탁매트", "수첩", "베개")
    private val testProducts = listOf("여행파우치", "보온병", "슬리퍼", "우비", "스탠드", "담요", "달력", "프라이팬", "선반", "도시락통",
        "양말", "케이블", "목도리", "스피커", "접시", "커튼", "지갑", "행주", "필통", "방석")
    private val topics = listOf("여행", "주방", "의류", "여행", "전자기기", "생활", "문구", "주방", "가구", "주방",
        "의류", "전자기기", "의류", "전자기기", "주방", "생활", "여행", "주방", "문구", "생활")

    fun items(dataset: String): List<LifeSyntheticItem> {
        require(dataset == DEV || dataset == TEST)
        val dev = dataset == DEV
        val products = if (dev) devProducts else testProducts
        val prefix = if (dev) "DEV" else "TST"
        return products.flatMapIndexed { index, product ->
            val merchant = merchants[index % merchants.size]
            val order = prefix + (1000 + index)
            val topic = topics[index]
            val first = if (dev) product + " 주문을 확인하고 정리했다." else product + "을 구입한 기록이다."
            val second = if (dev) product + " 배송 예정 안내" else product + " 택배가 출발했다."
            val third = if (dev) product + " 반품 신청은 2030년 11월 20일까지." else product + " 교환 신청은 2031년 12월 15일까지."
            listOf(
                LifeSyntheticItem(order + "-1", order, topic, product + " 주문", "상점: " + merchant + "\n주문번호: " + order + "\n" + first, merchant),
                LifeSyntheticItem(order + "-2", order, topic, "배송 안내", "상점: " + merchant + "\n주문번호: " + order + "\n" + second, merchant),
                LifeSyntheticItem(order + "-3", order, topic, product + " 메모", "상점: " + merchant + "\n" + third, merchant),
            )
        }
    }

    fun orders(dataset: String): Map<String, List<LifeSyntheticItem>> = items(dataset).groupBy(LifeSyntheticItem::orderId)
}
