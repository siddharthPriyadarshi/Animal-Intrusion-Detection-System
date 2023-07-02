import React from 'react'
import { useState, useEffect } from 'react'
import { getAlertData } from '../services/getAlert'
import './TextScale.css'

const AlertView = () => {
  const [isAlert, setIsAlert] = useState(false)
  // const [count, setCount] = useState(0)

  useEffect(() => {
    const alertInterval = setInterval(() => {
      getAlertData().then((data) => {
        console.log('Data fetched', data.status)
        if (data.status === 200) {
          // if (count === 0) {
          //   setCount(1)
          //   clearInterval(alertInterval)
          // }
          setIsAlert(true)
        } else {
          setIsAlert(false)
        }
      })
      return(()=>{clearInterval(alertInterval)})
    }, 1000)
  }, [isAlert])

  return (
    <>
      <div>
        {isAlert ? (
          <h3 style={{ color: 'red', fontSize: '5rem' }} className="text-scale">
            ALERT 🔴
          </h3>
        ) : (
          <h3 style={{ color: 'green', fontSize: '5rem', textAlign: 'center' }}>
            No Alert 🟢
          </h3>
        )}
      </div>
    </>
  )
}
export default AlertView
