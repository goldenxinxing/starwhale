import { useState, useEffect } from 'react'
import { StatefulMenu } from 'baseui/menu'
import { PLACEMENT, Popover } from 'baseui/popover'
import { FilterT } from './types'
// eslint-disable-next-line import/no-cycle
import { useStyles } from './Search'
// eslint-disable-next-line import/no-cycle
import { Operators } from './utils'

function PopoverContainer(props: {
    onItemSelect?: (props: { item: { label: string; type: string } }) => void
    options: any[]
    isOpen: boolean
    mountNode?: HTMLElement
    children: React.ReactNode
    innerRef?: React.RefObject<HTMLElement>
}) {
    const [isOpen, setIsOpen] = useState(false)
    const styles = useStyles()

    useEffect(() => {
        setIsOpen(props.isOpen)
    }, [props.isOpen])

    const handleClose = () => setIsOpen(false)

    return (
        <Popover
            returnFocus
            autoFocus
            placement={PLACEMENT.bottomLeft}
            isOpen={isOpen}
            ignoreBoundary
            mountNode={props.mountNode}
            content={() => (
                <StatefulMenu
                    rootRef={props.innerRef}
                    items={props.options}
                    onItemSelect={(args) => {
                        // @ts-ignore
                        props.onItemSelect?.(args)
                        handleClose()
                    }}
                    overrides={{
                        List: {
                            style: { minHeight: '150px', minWidth: '150px', maxHeight: '500px', overflow: 'auto' },
                        },
                    }}
                />
            )}
        >
            <p
                className={props.children ? styles.label : ''}
                title={typeof props.children === 'string' ? props.children : ''}
            >
                {props.children}
            </p>
        </Popover>
    )
}

function Filter(options: FilterT): FilterT {
    const operatorOptions = options.operators.map((key: string) => {
        const operator = Operators[key]
        return {
            id: operator.key,
            type: operator.key,
            label: operator.label,
        }
    })

    return {
        kind: options.kind,
        operators: options.operators,
        renderField: function RenderField({ options: renderOptions = [], isEditing = false, ...rest }) {
            return (
                <PopoverContainer
                    {...rest}
                    options={renderOptions}
                    isOpen={isEditing}
                    onItemSelect={({ item }) => rest.onChange?.(item.type)}
                >
                    {rest.value}
                </PopoverContainer>
            )
        },
        renderOperator: function RenderOperator({ isEditing = false, ...rest }) {
            return (
                <PopoverContainer
                    {...rest}
                    options={operatorOptions}
                    isOpen={isEditing}
                    onItemSelect={({ item }) => rest.onChange?.(item.type)}
                >
                    {operatorOptions.find((v) => v.type === rest.value)?.label ?? ''}
                </PopoverContainer>
            )
        },
        // renderFieldValue: options?.renderFieldValue ?? <></>,
    }
}
export default Filter
