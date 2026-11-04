package com.heecchoi.slmwikipoc

import android.content.Context

object LifeServices {
    private var sharedStore: LocalStore? = null
    private var sharedModel: EmbeddingModel? = null
    private var sharedRepository: LifeRepository? = null

    @Synchronized fun store(context: Context): LocalStore =
        sharedStore ?: LocalStore(context.applicationContext).also { sharedStore = it }

    @Synchronized fun model(context: Context): EmbeddingModel =
        sharedModel ?: EmbeddingModel.from(context.applicationContext).also { sharedModel = it }

    @Synchronized fun repository(context: Context): LifeRepository =
        sharedRepository ?: LifeRepository(context.applicationContext).also { sharedRepository = it }
}
