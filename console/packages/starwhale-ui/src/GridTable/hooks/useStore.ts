import { useContext, useMemo } from 'react'
import { useStore as useZustandStore } from 'zustand'
import type { StoreApi } from 'zustand'

import StoreContext from '../contexts/GridStoreContext'
import { IGridState } from '../types'

const zustandErrorMessage = 'Could not find zustand store context value.'

type ExtractState = StoreApi<IGridState> extends { getState: () => infer T } ? T : never

function useStore<StateSlice = ExtractState>(
    selector: (state: IGridState) => StateSlice,
    equalityFn?: (a: StateSlice, b: StateSlice) => boolean
) {
    const store = useContext(StoreContext)

    if (store === null) {
        throw new Error(zustandErrorMessage)
    }

    return useZustandStore(store, selector, equalityFn)
}

const useStoreApi = () => {
    const store = useContext(StoreContext)

    if (store === null) {
        throw new Error(zustandErrorMessage)
    }

    return useMemo(
        () => ({
            getState: store.getState,
            setState: store.setState,
            subscribe: store.subscribe,
            destroy: store.destroy,
        }),
        [store]
    )
}

export { useStore, useStoreApi }
